from typing import Callable, Optional, Union
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3Attention, Qwen3MLP
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers.processing_utils import Unpack
from transformers.modeling_layers import GradientCheckpointingLayer
import os

# Environment variable to control debug output
DEBUG_FLEX_ATTENTION = os.getenv('DEBUG_FLEX_ATTENTION', '1') == '1'

# FlexAttention import
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("FlexAttention not available. Please ensure you have PyTorch 2.5+ with FlexAttention support.")

def debug_print(*args, **kwargs):
    """Conditional debug printing based on environment variable"""
    if DEBUG_FLEX_ATTENTION:
        print(*args, **kwargs)

def check_tensor_properties(tensor, name):
    """Debug helper to check tensor properties that might affect FlexAttention compilation"""
    debug_print(f"{name} properties:")
    debug_print(f"  shape: {tensor.shape}")
    debug_print(f"  dtype: {tensor.dtype}")
    debug_print(f"  device: {tensor.device}")
    debug_print(f"  requires_grad: {tensor.requires_grad}")
    debug_print(f"  is_contiguous: {tensor.is_contiguous()}")
    debug_print(f"  stride: {tensor.stride()}")
    debug_print(f"  memory_format: {tensor.memory_format if hasattr(tensor, 'memory_format') else 'N/A'}")

def check_system_state():
    """Check system state that might affect FlexAttention compilation"""
    debug_print("System state check:")
    debug_print(f"  PyTorch version: {torch.__version__}")
    debug_print(f"  CUDA available: {torch.cuda.is_available()}")
    debug_print(f"  Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    debug_print(f"  Is compiled: {hasattr(torch, '_C') and hasattr(torch._C, '_get_current_static_runtime')}")
    debug_print(f"  Gradient enabled: {torch.is_grad_enabled()}")
    debug_print(f"  In inference mode: {torch.is_inference_mode_enabled()}")
    
    # Check if we're in a problematic context
    try:
        # Try to access current patcher state
        import torch.fx._symbolic_trace as st
        debug_print(f"  Current patcher: {getattr(st, 'CURRENT_PATCHER', 'Not available')}")
    except Exception as e:
        debug_print(f"  Patcher check failed: {e}")

def create_position_embeddings_custom(seq_len, head_dim, device, dtype=torch.float32):
    """Create RoPE position embeddings for the given sequence length"""
    position_ids = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)
    
    # Create frequency matrix
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    
    # Compute position embeddings
    # Ensure position_ids remains 1-D after squeeze to avoid 0-D tensor issue
    position_ids_1d = position_ids.squeeze()
    if position_ids_1d.dim() == 0:
        position_ids_1d = position_ids_1d.unsqueeze(0)
    freqs = torch.outer(position_ids_1d, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Duplicate each frequency to match head_dim
    cos = torch.cat([cos, cos], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([sin, sin], dim=-1)  # (seq_len, head_dim)
    
    # Add batch dimension
    cos = cos.unsqueeze(0)  # (1, seq_len, head_dim)
    sin = sin.unsqueeze(0)  # (1, seq_len, head_dim)
    
    return cos, sin
class CompressedDecoderLayer(GradientCheckpointingLayer):
    """
    A decoder layer with optional compression capability.
    
    Args:
        enable_compression (bool): Whether to enable compression. When False, 
                                 behaves like a regular decoder layer without compression.
                                 Default: True
    """
    def __init__(self, config: Qwen3Config, layer_idx: int, T_w: int , r: float , M: Optional[int] = None, enable_compression: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.enable_compression = enable_compression

        self.self_attn = CompressedAttention(layer_idx, T_w, r, config, M, enable_compression=enable_compression)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = getattr(config, 'layer_types', ['attention'] * getattr(config, 'num_hidden_layers', 32))[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        compressed_attention_mask: Optional[torch.Tensor] = None,  # New parameter for compressed attention mask from previous layer
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        # Store original dtype
        original_dtype = hidden_states.dtype
        
        residual = hidden_states
        
        # Apply layer norm (may need specific dtype)
        if self.input_layernorm.weight.dtype != hidden_states.dtype:
            hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - handle the 5-tuple return from CompressedAttention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=compressed_attention_mask or attention_mask,  # Use compressed mask if available
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Unpack the 5-tuple from CompressedAttention
        residual_from_attn, hidden_states, self_attn_weights, present_key_value, new_compressed_attention_mask = attn_output
        
        # Add residuals (both should be in original_dtype from CompressedAttention)
        hidden_states = residual_from_attn + hidden_states

        # Fully Connected
        residual = hidden_states
        
        # Apply layer norm (may need specific dtype)
        if self.post_attention_layernorm.weight.dtype != hidden_states.dtype:
            hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP (may need specific dtype)
        if self.mlp.gate_proj.weight.dtype != hidden_states.dtype:
            hidden_states = self.mlp(hidden_states.to(dtype=self.mlp.gate_proj.weight.dtype))
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        # Return format compatible with unsloth: return tensor when single output, tuple for multiple
        if output_attentions or use_cache:
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            # Include the compressed attention mask for the next layer (may be None if compression disabled)
            outputs += (new_compressed_attention_mask,)
            return outputs
        else:
            # Return both hidden states and compressed attention mask for next layer
            # Always return the compressed attention mask to maintain sequence length consistency
            return hidden_states, new_compressed_attention_mask


class CompressedAttention(nn.Module):
    """Multi-headed attention with optional compression from Qwen3 model
    
    Args:
        enable_compression (bool): Whether to enable compression. When False,
                                 behaves like regular attention without compression.
                                 Default: True
    """

    def __init__(
        self, 
        layer_idx: int,
        T_w: int , 
        r: float , 
        config: Qwen3Config,
        M: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_compression: bool = True
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        self.compress = MLPCompression(config)
        self.T_w = T_w
        self.r = r
        self.M = M
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if hasattr(config, 'layer_types') and config.layer_types[layer_idx] == "sliding_attention" else None
        self.enable_compression = enable_compression
        
        # FlexAttention setup
        self.use_flex_attention = FLEX_ATTENTION_AVAILABLE and getattr(config, 'use_flex_attention', True)

    def create_causal_attention_mask_mod(self, seq_len: int):
        """Create a mask_mod function for causal attention using FlexAttention"""
        def causal_mask(b, h, q_idx, kv_idx):
            # Causal mask: query can only attend to positions <= its position
            return q_idx >= kv_idx
        
        return causal_mask

    def create_causal_block_mask(self, seq_len: int, device: torch.device):
        """Create a causal BlockMask for any sequence length (compressed or uncompressed)"""
        if not self.use_flex_attention:
            return None
            
        try:
            mask_mod = self.create_causal_attention_mask_mod(seq_len)
            
            # Create block mask for the compressed sequence
            # Set B=None, H=None for broadcasting since causal pattern is the same across batch/heads
            # Use device string format that's most compatible
            device_str = str(device) if isinstance(device, torch.device) else device
            
            block_mask = create_block_mask(
                mask_mod, 
                B=None, 
                H=None, 
                Q_LEN=seq_len, 
                KV_LEN=seq_len,
                device=device_str
            )
            
            return block_mask
            
        except Exception as e:
            debug_print(f"Block mask creation failed: {e}")
            # Return None if block mask creation fails - will use minimal FlexAttention
            return None

    def compress_memory(self, x: torch.Tensor, T_w: int, r: float , M: Optional[int] = None):
        """Compress memory part of the sequence"""
        B, T, C = x.shape
        
        # Handle odd sequence length
        if T % 2 == 1:
            T_w = T_w + 1
        if T_w >= T:
            return x, None, x  # Return original if no compression needed
            
        # Split into memory and window parts
        x_m = x[:, :-T_w]
        x_w = x[:, -T_w:]
        
        # Compress memory part - MLPCompression handles dtype consistency internally
        xm_cmp = self.compress(x_m)
        
        return x_m, xm_cmp, x_w

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]], Optional[torch.Tensor]]:

        
        # Store original dtype and establish working dtype
        original_dtype = hidden_states.dtype
        working_dtype = self.q_proj.weight.dtype
        
        # Convert to working dtype once at the beginning
        hidden_states = hidden_states.to(dtype=working_dtype)
        
        ### Compression of memory part ###
        if self.enable_compression:
            x_m, xm_cmp, x_w = self.compress_memory(hidden_states, self.T_w, self.r, self.M)
            
            if xm_cmp is None:
                hidden_states = x_w
                residuals = hidden_states.clone()
            else:
                # Create proper query and key representations for final compression
                B, T_w_actual, C = x_w.shape
                B_m, T_m, C_m = xm_cmp.shape
                
                # No dtype conversion needed - already in working_dtype
                q_w_proj = self.q_proj(x_w).view(B, self.config.num_attention_heads, T_w_actual, self.head_dim)
                q_w = self.q_norm(q_w_proj)
                
                km_cmp_proj = self.k_proj(xm_cmp).view(B_m, self.config.num_key_value_heads, T_m, self.head_dim)
                km_cmp = self.k_norm(km_cmp_proj)
                
                FinalCompression = FinalCompressedTokens(self.config, q_w, km_cmp, self.r, self.M)
                compressed_tokens = FinalCompression(x_m, xm_cmp)
                
                hidden_states = torch.cat([compressed_tokens, x_w], dim=1)
                residuals = hidden_states.clone()
        else:
            # Skip compression - use original hidden states directly
            residuals = hidden_states.clone()
        
        # Update shapes after compression
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Main attention projections - already in working_dtype
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        

        # Create position embeddings for the compressed sequence
        seq_len_compressed = hidden_states.shape[1]
        position_ids = torch.arange(seq_len_compressed, device=hidden_states.device).unsqueeze(0)
        position_embeddings = create_position_embeddings_custom(seq_len_compressed, self.head_dim, hidden_states.device, working_dtype)
        cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        
        # Create attention mask for the sequence (compressed or original)
        seq_len_current = hidden_states.shape[1]
        
        if self.enable_compression:
            # Always create a new attention mask that matches the compressed sequence length
            compressed_attention_mask = torch.triu(torch.ones(seq_len_current, seq_len_current, device=hidden_states.device, dtype=working_dtype), diagonal=1)
            compressed_attention_mask = compressed_attention_mask.masked_fill(compressed_attention_mask == 1, -float('inf'))
            compressed_attention_mask = compressed_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_current, seq_len_current)
            
            # Update the attention_mask parameter to match compressed sequence length
            attention_mask = compressed_attention_mask
        else:
            # When compression is disabled, use original attention_mask if provided, 
            # otherwise create a causal mask for the original sequence
            if attention_mask is None:
                attention_mask = torch.triu(torch.ones(seq_len_current, seq_len_current, device=hidden_states.device, dtype=working_dtype), diagonal=1)
                attention_mask = attention_mask.masked_fill(attention_mask == 1, -float('inf'))
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_current, seq_len_current)
            else:
                # Ensure existing attention_mask has the correct dtype and shape
                attention_mask = attention_mask.to(dtype=working_dtype)
                # Handle different attention mask formats
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
        

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Try FlexAttention first, fallback to standard attention if it fails
        attn_output = None
        attn_weights = None
        
        if self.use_flex_attention:
            try:
                # Prepare tensors for FlexAttention - already in working_dtype and contiguous
                query_states_flex = query_states.contiguous()
                key_states_flex = key_states.contiguous()
                value_states_flex = value_states.contiguous()
                

                # Handle grouped query attention
                if self.num_key_value_groups > 1:
                    key_states_flex = key_states_flex.repeat_interleave(self.num_key_value_groups, dim=1)
                    value_states_flex = value_states_flex.repeat_interleave(self.num_key_value_groups, dim=1)

                # Create block mask for the current sequence length (compressed or uncompressed)
                block_mask = self.create_causal_block_mask(seq_len_current, hidden_states.device)
                
                # Try FlexAttention with block mask
                if block_mask is not None:
                    flex_output = flex_attention(query_states_flex, key_states_flex, value_states_flex, block_mask=block_mask)
                else:
                    # Fallback to FlexAttention without block mask
                    flex_output = flex_attention(query_states_flex, key_states_flex, value_states_flex)
                
                # Handle FlexAttention output (might be a tuple)
                if isinstance(flex_output, tuple):
                    attn_output = flex_output[0]  # Take the attention output
                else:
                    attn_output = flex_output
                    
                attn_weights = None
                
            except Exception as e:
                # FlexAttention failed, will use standard attention fallback
                attn_output = None
        
        if attn_output is None:
            # Fallback to standard attention - already in working_dtype
            query_states_std = query_states.contiguous()
            key_states_std = key_states.contiguous()
            value_states_std = value_states.contiguous()
            
            
            # Handle grouped query attention
            if self.num_key_value_groups > 1:
                batch_size, num_kv_heads, seq_len, head_dim = key_states_std.shape
                target_num_heads = num_kv_heads * self.num_key_value_groups
                
                # Expand key/value states to match query heads
                key_states_std = key_states_std.unsqueeze(2).expand(batch_size, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
                key_states_std = key_states_std.reshape(batch_size, target_num_heads, seq_len, head_dim)
                
                value_states_std = value_states_std.unsqueeze(2).expand(batch_size, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
                value_states_std = value_states_std.reshape(batch_size, target_num_heads, seq_len, head_dim)
                
            
            # Standard scaled dot-product attention
            scores = torch.matmul(query_states_std, key_states_std.transpose(-2, -1)) * self.scaling
            
            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                scores = scores + attention_mask.to(dtype=scores.dtype)
            
            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            
            if self.training and self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            
            # Compute output
            attn_output = torch.matmul(attn_weights, value_states_std)

            
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        # Output projection - already in working_dtype
        attn_output = self.o_proj(attn_output)
        
        # Convert back to original dtype only at the end
        attn_output = attn_output.to(dtype=original_dtype)
        residuals = residuals.to(dtype=original_dtype)
        
        # Return appropriate attention mask based on compression state
        if self.enable_compression:
            if self.use_flex_attention:
                # For FlexAttention, return sequence length info for next layer's block mask creation
                output_attention_mask = torch.tensor([seq_len_current], 
                                                   device=hidden_states.device, 
                                                   dtype=torch.long)
            else:
                # For standard attention, return the actual attention mask
                output_attention_mask = compressed_attention_mask
        else:
            output_attention_mask = None
            
        return residuals, attn_output, attn_weights, None, output_attention_mask  # Return 5-tuple with compressed attention mask

class NoLoRALinear(nn.Linear):
    """A Linear layer that prevents LoRA adaptation by overriding __getattr__"""
    
    def __getattr__(self, name):
        # Prevent LoRA attribute access
        if name in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B', 'scaling']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' (LoRA not supported)")
        return super().__getattr__(name)

class MLPCompression(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Use NoLoRALinear to prevent LoRA application
        self.gate_proj = NoLoRALinear(2*self.hidden_size, int(self.intermediate_size/4), bias=False)
        self.up_proj = NoLoRALinear(2*self.hidden_size, int(self.intermediate_size/4), bias=False)
        self.down_proj = NoLoRALinear(int(self.intermediate_size/4), self.hidden_size, bias=False)
        
        # Get the activation function - ACT2FN returns instantiated functions
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        B, T, C = x.shape
        T_even = T - (T % 2)  # Handle odd sequence length
        
        if T_even > 0:
            # Reshape to get pairs of tokens
            x_pairs = x[:, :T_even].view(B, T_even // 2, 2 * C)
            
            # Compute gates and transform dtype only if needed
            gate_pairs = x_pairs if x_pairs.dtype == self.gate_proj.weight.dtype else x_pairs.to(self.gate_proj.weight.dtype)
            up_pairs = x_pairs if x_pairs.dtype == self.up_proj.weight.dtype else x_pairs.to(self.up_proj.weight.dtype)
            
            gate_out = self.gate_proj(gate_pairs)
            up_out = self.up_proj(up_pairs)
            
            # Convert back to x.dtype only if different
            gate_out = gate_out if gate_out.dtype == x.dtype else gate_out.to(x.dtype)
            up_out = up_out if up_out.dtype == x.dtype else up_out.to(x.dtype)
            
            # Now apply gates and continue processing
            mlp_out = self.down_proj(self.act_fn(gate_out) * up_out)
            mlp_out = mlp_out if mlp_out.dtype == x.dtype else mlp_out.to(x.dtype)
            
            # Apply activation and set output
            mlp_out = self.act_fn(mlp_out)
            x_compressed = mlp_out
        else:
            x_compressed = torch.empty(B, 0, C, device=x.device, dtype=x.dtype)
        
        return x_compressed

# class MLPCompression(nn.Module):
#     def __init__(self, config: Qwen3Config)
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         # Use NoLoRALinear to prevent LoRA application
#         self.gate_proj = NoLoRALinear(2*self.hidden_size, int(self.intermediate_size/4), bias=False)
#         self.up_proj = NoLoRALinear(2*self.hidden_size, int(self.intermediate_size/4), bias=False)
#         self.down_proj = NoLoRALinear(int(self.intermediate_size/4), self.hidden_size, bias=False)

#         # FIXED intermediate size 10

#         # self.gate_proj = NoLoRALinear(2*self.hidden_size, 10, bias=False)
#         # self.up_proj = NoLoRALinear(2*self.hidden_size, 10, bias=False)
#         # self.down_proj = NoLoRALinear(10, self.hidden_size, bias=False)
        
#         # Initialize gate projections to zero
#         # nn.init.zeros_(self.gate_proj.weight)
#         # nn.init.zeros_(self.up_proj.weight)
#         # nn.init.zeros_(self.down_proj.weight)
        
#         # Get the activation function - ACT2FN returns instantiated functions
#         self.act_fn = ACT2FN[config.hidden_act]
        
#         # Learnable parameters for weighted combination
#         self.a = nn.Parameter(torch.ones(1)/2)
#         self.b = nn.Parameter(torch.ones(1)/2)
        
#         # Normalization layer after compression
#         self.norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

#     def forward(self, x):
#         B, T, C = x.shape
#         T_even = T - (T % 2)
        
        
#         # Store original dtype and establish working dtype
#         original_dtype = x.dtype
#         working_dtype = self.gate_proj.weight.dtype
        
#         # Convert to working dtype once at the beginning
#         x = x.to(dtype=working_dtype)
        
#         if T_even > 0:
#             # Reshape to get pairs
#             x_pairs = x[:, :T_even].view(B, T_even // 2, 2 * C)
            
#             # Extract individual tokens from pairs
#             x_even = x[:, :T_even:2]  # x[2*i] - tokens at even positions
#             x_odd = x[:, 1:T_even:2]  # x[2*i+1] - tokens at odd positions
            
#             # All tensors are already in working_dtype
#             # Ensure a and b parameters are in working_dtype
#             a = self.a.to(dtype=working_dtype)
#             b = self.b.to(dtype=working_dtype)
            
#             # Compute MLP component
#             gate_out = self.gate_proj(x_pairs)
#             up_out = self.up_proj(x_pairs)
            
#             mlp_out = self.down_proj(self.act_fn(gate_out) * up_out)
            
#             # Compute weighted combination: a * x[2*i] + b * x[2*i+1] + MLP(x_pairs)
#             x_compressed = a * x_even + b * x_odd + mlp_out
#             x_compressed = self.act_fn(x_compressed)
            
#             # Apply normalization after compression
#             x_compressed = self.norm(x_compressed)
            
#         else:
#             x_compressed = torch.empty(B, 0, C, device=x.device, dtype=working_dtype)
        
#         # Convert back to original dtype only at the end
#         return x_compressed.to(dtype=original_dtype)

class FinalCompressedTokens(nn.Module):
    def __init__(self, config: Qwen3Config, q_w: torch.Tensor, km_cmp: torch.Tensor, r: float, M: Optional[int] = None):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_w = q_w
        self.km_cmp = km_cmp
        self.r = r
        self.M = M

    def importance_score(self):
        """Compute importance scores by calculating attention between window queries and memory keys."""
        q_w = self.q_w
        km_cmp = self.km_cmp
        
        
        # Ensure dtype consistency - use query dtype as working dtype
        working_dtype = q_w.dtype
        km_cmp = km_cmp.to(dtype=working_dtype)
        
        # Handle grouped query attention: repeat key heads to match query heads
        if self.num_key_value_groups > 1:
            # Ensure tensor is contiguous before expansion
            km_cmp = km_cmp.contiguous()
            
            # Use expand instead of repeat_interleave for better compatibility
            batch_size, num_kv_heads, seq_len, head_dim = km_cmp.shape
            target_num_heads = num_kv_heads * self.num_key_value_groups
            
            
            # Expand km_cmp: (B, num_kv_heads, seq_len, head_dim) -> (B, num_heads, seq_len, head_dim)
            km_cmp = km_cmp.unsqueeze(2).expand(batch_size, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
            km_cmp = km_cmp.reshape(batch_size, target_num_heads, seq_len, head_dim)
            
        # Compute attention scores
        
        attention_scores = torch.matmul(q_w, km_cmp.transpose(-2, -1)) * self.scaling
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        if self.training and self.attention_dropout > 0:
            attention_weights = F.dropout(attention_weights, p=self.attention_dropout)
        
        # Compute importance scores by summing over query positions and averaging over heads
        cumulative_attention = attention_weights.sum(dim=2)  # Sum over query positions: (B, num_heads, T_m)
        
        importance_scores = cumulative_attention.mean(dim=1)  # Average over heads: (B, T_m)
        
        return importance_scores

    def forward(self, x_m: torch.Tensor, xm_cmp: torch.Tensor) -> torch.Tensor:
        # Store original dtype
        original_dtype = x_m.dtype
        
        importance_scores = self.importance_score()
        
        indices = select_indices(importance_scores, self.r, self.M)
        
        x_m_expanded = interleave_selected(x_m, xm_cmp, indices)
        
        # Ensure output has the correct dtype
        return x_m_expanded.to(dtype=original_dtype)

        
def select_indices(importance_scores: torch.Tensor, r: float, M: Optional[int] = None):
    B, T_m = importance_scores.shape
    if M is None:
        M = 0
    num_to_select = int(r * T_m + M * (1 - r)/2)
    num_to_select = min(num_to_select, T_m)
    if num_to_select == 0:
        return torch.empty(B, 0, device=importance_scores.device, dtype=torch.long)
    selected_indices = torch.topk(importance_scores, num_to_select, dim=1).indices
    return selected_indices

def interleave_selected(x_m: torch.Tensor, xm_cmp: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    """Pure tensor implementation for interleaving selected tokens"""
    B, T_cmp, C = xm_cmp.shape
    device = xm_cmp.device
    
    # Use xm_cmp dtype as working dtype since it's the compressed representation
    working_dtype = xm_cmp.dtype
    
    # Convert x_m to working dtype if needed
    x_m = x_m.to(dtype=working_dtype)
    
    # Handle empty selection
    if selected_indices.numel() == 0:
        return xm_cmp

    # Handle single batch case
    if selected_indices.dim() == 1:
        selected_indices = selected_indices.unsqueeze(0).expand(B, -1)
    
    num_selected = selected_indices.shape[1]
    if num_selected == 0:
        return xm_cmp
        
    output_length = T_cmp + num_selected
    
    # Create expand mask
    expand_mask = torch.zeros(B, T_cmp, device=device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    expand_mask[batch_idx, selected_indices] = True
    
    # Get original token pairs for selected positions
    pair_start_indices = 2 * selected_indices
    pair_end_indices = 2 * selected_indices + 1
    
    # Check bounds
    T_orig = x_m.shape[1]
    if pair_end_indices.max() >= T_orig:
        return xm_cmp  # Return compressed if bounds exceeded
    
    batch_idx_pairs = batch_idx.expand(-1, num_selected)
    pair_starts = x_m[batch_idx_pairs, pair_start_indices]
    pair_ends = x_m[batch_idx_pairs, pair_end_indices]
    
    # Calculate positions
    position_sizes = torch.where(expand_mask, 2, 1)
    cum_positions = torch.cumsum(position_sizes, dim=1)
    start_positions = cum_positions - position_sizes
    
    # Initialize output with working dtype
    y = torch.zeros(B, output_length, C, device=device, dtype=working_dtype)
    
    # Place kept tokens
    keep_mask = ~expand_mask
    if keep_mask.any():
        keep_start_pos = start_positions[keep_mask]
        keep_batch_idx = batch_idx.expand(-1, T_cmp)[keep_mask]
        keep_tokens = xm_cmp[keep_mask]
        y[keep_batch_idx, keep_start_pos] = keep_tokens
    
    # Place expanded tokens
    if expand_mask.any():
        expand_start_pos = start_positions[expand_mask]
        expand_batch_idx = batch_idx.expand(-1, T_cmp)[expand_mask]
        
        # Simple mapping for expanded positions
        expand_indices = torch.arange(num_selected, device=device).unsqueeze(0).expand(B, -1)
        expand_indices = expand_indices[expand_mask[batch_idx, selected_indices]]
        
        # Place expanded tokens
        pair_start_tokens = pair_starts[expand_batch_idx, expand_indices]
        pair_end_tokens = pair_ends[expand_batch_idx, expand_indices]
            
        y[expand_batch_idx, expand_start_pos] = pair_start_tokens
        y[expand_batch_idx, expand_start_pos + 1] = pair_end_tokens
    
    return y