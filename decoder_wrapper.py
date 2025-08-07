import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
import copy
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from Compress import CompressedDecoderLayer

class Qwen3Compressed(nn.Module):
    """
    A compressed version of Qwen3ForCausalLM that replaces ALL layers with CompressedDecoderLayer
    but only enables compression for specified layers.
    
    This approach ensures all layers have the same structure while allowing selective compression.
    """
    
    def __init__(
        self,
        original_model: Qwen3ForCausalLM,
        compressed_layer_indices: Union[int, List[int]],
        T_w: int,
        r: float,
        M: Optional[int] = None
    ):
        """
        Initialize the Qwen3Compressed model.
        
        Args:
            original_model: The original Qwen3ForCausalLM model
            compressed_layer_indices: Index or list of indices of layers to enable compression for
            T_w: Window size for compression
            r: Compression ratio parameter
            M: Optional parameter for final compression
        """
        super().__init__()
        
        # Handle both single int and list of ints for compressed_layer_indices
        if isinstance(compressed_layer_indices, int):
            self.compressed_layer_indices = [compressed_layer_indices]
        else:
            self.compressed_layer_indices = compressed_layer_indices.copy()
            
        self.T_w = T_w
        self.r = r
        self.M = M
        
        # Validate layer indices
        num_layers = len(original_model.model.layers)
        for idx in self.compressed_layer_indices:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})")
        
        # Store the original model structure
        self.model: Qwen3ForCausalLM = copy.deepcopy(original_model)
        
        # Get the config and ensure it's the correct type
        config = original_model.config
        if not isinstance(config, Qwen3Config):
            # Create a Qwen3Config from the original config if needed
            config = Qwen3Config(**config.__dict__)
        
        # Replace ALL decoder layers with CompressedDecoderLayer
        for layer_idx in range(num_layers):
            # Get the original decoder layer
            original_layer = self.model.model.layers[layer_idx]
            
            # Determine if compression should be enabled for this layer
            enable_compression = layer_idx in self.compressed_layer_indices
            
            # Create the compressed decoder layer (compression enabled or disabled)
            compressed_layer = CompressedDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                T_w=T_w,
                r=r,
                M=M,
                enable_compression=enable_compression
            )
        
            # Copy pretrained weights from original layer
            self._copy_pretrained_weights(original_layer, compressed_layer, layer_idx)
        
            # Replace the decoder layer
            self.model.model.layers[layer_idx] = compressed_layer
        
        # Move to the same device as the original model (skip for quantized models)
        if not self._is_quantized_model(original_model):
            try:
                original_device = next(original_model.parameters()).device
                # Use type ignore to suppress linter confusion about .to() method
                self.model = self.model.to(original_device)  # type: ignore
            except (ValueError, RuntimeError) as e:
                # Handle case where .to() is not supported (e.g., quantized models)
                if "not supported" in str(e):
                    pass  # Skip device movement for quantized models
                else:
                    raise
    
    def _copy_pretrained_weights(self, original_layer, compressed_layer, layer_idx: int):
        """
        Copy compatible pretrained weights from original Qwen3DecoderLayer to CompressedDecoderLayer.
        
        Args:
            original_layer: The original Qwen3DecoderLayer
            compressed_layer: The new CompressedDecoderLayer
            layer_idx: The layer index being processed
        """
        
        # Copy layer normalization weights
        if hasattr(original_layer, 'input_layernorm') and hasattr(compressed_layer, 'input_layernorm'):
            compressed_layer.input_layernorm.weight.data.copy_(original_layer.input_layernorm.weight.data)
        
        if hasattr(original_layer, 'post_attention_layernorm') and hasattr(compressed_layer, 'post_attention_layernorm'):
            compressed_layer.post_attention_layernorm.weight.data.copy_(original_layer.post_attention_layernorm.weight.data)
        
        # Copy attention weights from original attention to compressed attention
        if hasattr(original_layer, 'self_attn') and hasattr(compressed_layer, 'self_attn'):
            original_attention = original_layer.self_attn
            compressed_attention = compressed_layer.self_attn
            
            # Copy query projection weights
            if hasattr(original_attention, 'q_proj') and hasattr(compressed_attention, 'q_proj'):
                compressed_attention.q_proj.weight.data.copy_(original_attention.q_proj.weight.data)
                if original_attention.q_proj.bias is not None and compressed_attention.q_proj.bias is not None:
                    compressed_attention.q_proj.bias.data.copy_(original_attention.q_proj.bias.data)
            
            # Copy key projection weights
            if hasattr(original_attention, 'k_proj') and hasattr(compressed_attention, 'k_proj'):
                compressed_attention.k_proj.weight.data.copy_(original_attention.k_proj.weight.data)
                if original_attention.k_proj.bias is not None and compressed_attention.k_proj.bias is not None:
                    compressed_attention.k_proj.bias.data.copy_(original_attention.k_proj.bias.data)
            
            # Copy value projection weights
            if hasattr(original_attention, 'v_proj') and hasattr(compressed_attention, 'v_proj'):
                compressed_attention.v_proj.weight.data.copy_(original_attention.v_proj.weight.data)
                if original_attention.v_proj.bias is not None and compressed_attention.v_proj.bias is not None:
                    compressed_attention.v_proj.bias.data.copy_(original_attention.v_proj.bias.data)
            
            # Copy output projection weights
            if hasattr(original_attention, 'o_proj') and hasattr(compressed_attention, 'o_proj'):
                compressed_attention.o_proj.weight.data.copy_(original_attention.o_proj.weight.data)
                if original_attention.o_proj.bias is not None and compressed_attention.o_proj.bias is not None:
                    compressed_attention.o_proj.bias.data.copy_(original_attention.o_proj.bias.data)
            
            # Copy query normalization weights
            if hasattr(original_attention, 'q_norm') and hasattr(compressed_attention, 'q_norm'):
                compressed_attention.q_norm.weight.data.copy_(original_attention.q_norm.weight.data)
            
            # Copy key normalization weights
            if hasattr(original_attention, 'k_norm') and hasattr(compressed_attention, 'k_norm'):
                compressed_attention.k_norm.weight.data.copy_(original_attention.k_norm.weight.data)

        # ------------------------------------------------------------------
        # Copy MLP (feed-forward) weights – these are critical for preserving
        # the original model’s behaviour but were previously omitted.
        # ------------------------------------------------------------------
        if hasattr(original_layer, 'mlp') and hasattr(compressed_layer, 'mlp'):
            orig_mlp = original_layer.mlp
            comp_mlp = compressed_layer.mlp

            def _safe_copy(attr_name):
                if hasattr(orig_mlp, attr_name) and hasattr(comp_mlp, attr_name):
                    orig_lin = getattr(orig_mlp, attr_name)
                    comp_lin = getattr(comp_mlp, attr_name)
                    # Weight
                    comp_lin.weight.data.copy_(orig_lin.weight.data)
                    # Bias (some implementations use bias=False)
                    if orig_lin.bias is not None and comp_lin.bias is not None:
                        comp_lin.bias.data.copy_(orig_lin.bias.data)

            for proj_name in ['gate_proj', 'up_proj', 'down_proj', 'out_proj']:
                _safe_copy(proj_name)

            # If the MLP uses a norm layer (e.g. RMSNorm) copy its weight too
            if hasattr(orig_mlp, 'act_fn') and hasattr(comp_mlp, 'act_fn'):
                pass  # activation functions are stateless – nothing to copy

            if hasattr(orig_mlp, 'norm') and hasattr(comp_mlp, 'norm'):
                if hasattr(orig_mlp.norm, 'weight') and hasattr(comp_mlp.norm, 'weight'):
                    comp_mlp.norm.weight.data.copy_(orig_mlp.norm.weight.data)
        
        # Note: The compression-specific components (compress module) will use their initialized weights
    
    def _is_quantized_model(self, model) -> bool:
        """Check if the model is quantized (8-bit or 4-bit)."""
        # Check for quantization config
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            return model.config.quantization_config is not None
        
        # Check for bitsandbytes quantization attributes
        if hasattr(model, 'is_quantized'):
            return model.is_quantized
            
        # Check if any parameter is quantized
        for param in model.parameters():
            if hasattr(param, 'CB') or hasattr(param, 'SCB'):  # bitsandbytes attributes
                return True
                
        return False

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        outputs = self.model(*args, **kwargs)
        return outputs
    
    def generate(self, *args, **kwargs):
        """Generation method for text generation."""
        if hasattr(self.model, 'generate'):
            # Handle unsloth compatibility issues
            # Ensure use_cache is set to False if not specified to avoid past_key_values issues
            if 'use_cache' not in kwargs:
                kwargs['use_cache'] = False
            
            # Explicitly set past_key_values to None if not provided
            if 'past_key_values' not in kwargs:
                kwargs['past_key_values'] = None
                
            # Add other helpful defaults for generation
            if 'do_sample' not in kwargs and 'temperature' in kwargs and kwargs['temperature'] > 0.0:
                kwargs['do_sample'] = True
                
            if 'pad_token_id' not in kwargs and hasattr(self.model, 'config') and hasattr(self.model.config, 'eos_token_id'):
                kwargs['pad_token_id'] = self.model.config.eos_token_id
            
            return self.model.generate(*args, **kwargs)
        else:
            raise AttributeError(f"Model {type(self.model)} does not have a generate method")
    
    def get_compressed_layers(self) -> Dict[int, Any]:
        """Get all compressed decoder layers."""
        compressed_layers = {}
        for layer_idx in self.compressed_layer_indices:
            compressed_layers[layer_idx] = self.model.model.layers[layer_idx]
        return compressed_layers
    
    def get_compressed_layer(self, layer_idx: Optional[int] = None):
        """
        Get a specific compressed decoder layer.
        
        Args:
            layer_idx: Index of the layer to get. If None and only one layer is compressed, 
                      returns that layer. Otherwise raises an error.
        """
        if layer_idx is not None:
            if layer_idx not in self.compressed_layer_indices:
                raise ValueError(f"Layer {layer_idx} is not a compressed layer. Compressed layers: {self.compressed_layer_indices}")
            return self.model.model.layers[layer_idx]
        else:
            if len(self.compressed_layer_indices) == 1:
                return self.model.model.layers[self.compressed_layer_indices[0]]
            else:
                raise ValueError(f"Multiple compressed layers exist {self.compressed_layer_indices}. Please specify layer_idx.")
    
    def get_all_layers(self) -> Dict[int, Any]:
        """Get all decoder layers (both compressed and non-compressed)."""
        all_layers = {}
        for layer_idx in range(len(self.model.model.layers)):
            all_layers[layer_idx] = self.model.model.layers[layer_idx]
        return all_layers
    
    def get_compression_params(self) -> Dict[str, Any]:
        """Get the compression parameters used."""
        return {
            'compressed_layer_indices': self.compressed_layer_indices,
            'T_w': self.T_w,
            'r': self.r,
            'M': self.M
        }
    
    def unfreeze_layer_parameters(self, layer_idx: Optional[Union[int, List[int]]] = None):
        """
        Unfreeze all parameters at the specified layer(s).
        
        Args:
            layer_idx: Index or list of indices of the layer(s) to unfreeze. 
                      If None, unfreezes all compressed layers.
        """
        if layer_idx is None:
            target_indices = self.compressed_layer_indices
        elif isinstance(layer_idx, int):
            target_indices = [layer_idx]
        else:
            target_indices = layer_idx
        
        for target_layer_idx in target_indices:
            # Unfreeze all parameters in the target layer
            target_layer = self.model.model.layers[target_layer_idx]
            for param in target_layer.parameters():
                param.requires_grad = True
            print(f"Unfroze all parameters at layer {target_layer_idx}")
    
    def freeze_layer_parameters(self, layer_idx: Optional[Union[int, List[int]]] = None):
        """
        Freeze all parameters at the specified layer(s).
        
        Args:
            layer_idx: Index or list of indices of the layer(s) to freeze. 
                      If None, freezes all compressed layers.
        """
        if layer_idx is None:
            target_indices = self.compressed_layer_indices
        elif isinstance(layer_idx, int):
            target_indices = [layer_idx]
        else:
            target_indices = layer_idx
        
        for target_layer_idx in target_indices:
            # Freeze all parameters in the target layer
            target_layer = self.model.model.layers[target_layer_idx]
            for param in target_layer.parameters():
                param.requires_grad = False
            print(f"Froze all parameters at layer {target_layer_idx}")
    
    def get_layer_parameter_status(self, layer_idx: Optional[Union[int, List[int]]] = None) -> Dict[str, bool]:
        """
        Get the requires_grad status of all parameters at the specified layer(s).
        
        Args:
            layer_idx: Index or list of indices of the layer(s) to check. 
                      If None, checks all compressed layers.
            
        Returns:
            Dictionary mapping parameter names to their requires_grad status.
        """
        if layer_idx is None:
            target_indices = self.compressed_layer_indices
        elif isinstance(layer_idx, int):
            target_indices = [layer_idx]
        else:
            target_indices = layer_idx
        
        param_status = {}
        for target_layer_idx in target_indices:
            target_layer = self.model.model.layers[target_layer_idx]
            for name, param in target_layer.named_parameters():
                param_status[f"layer_{target_layer_idx}.{name}"] = param.requires_grad
        
        return param_status
    
    def freeze_pretrained(self):
        """
        Freeze all pretrained model weights while keeping the compression-specific 
        components (compress module) trainable for all compressed layers.
        
        This method directly freezes only the old/pretrained parameters,
        leaving compression-specific components unfrozen.
        """
        # Collect all compression-specific parameter IDs to avoid freezing them
        compression_param_ids = set()
        
        for layer_idx in self.compressed_layer_indices:
            compressed_layer = self.model.model.layers[layer_idx]
            
            # Collect compress module parameters
            if hasattr(compressed_layer, 'self_attn') and hasattr(compressed_layer.self_attn, 'compress'):
                compress_module = getattr(compressed_layer.self_attn, 'compress', None)
                if compress_module is not None and hasattr(compress_module, 'parameters'):
                    for param in compress_module.parameters():
                        compression_param_ids.add(id(param))
                    print(f"Found compression module parameters at layer {layer_idx}")
                else:
                    print(f"WARNING: Compression module found but has no parameters at layer {layer_idx}")
            else:
                print(f"WARNING: No compression module found at layer {layer_idx}")
                # Let's check what attributes exist
                if hasattr(compressed_layer, 'self_attn'):
                    print(f"Available attributes in self_attn: {[attr for attr in dir(compressed_layer.self_attn) if not attr.startswith('_')]}")
                    
                    # Check if compress module exists with different name (MLPCompression)
                    for attr_name in dir(compressed_layer.self_attn):
                        if not attr_name.startswith('_'):
                            attr_obj = getattr(compressed_layer.self_attn, attr_name)
                            if hasattr(attr_obj, 'gate_proj') and hasattr(attr_obj, 'up_proj') and hasattr(attr_obj, 'down_proj'):
                                # This looks like MLPCompression
                                for param in attr_obj.parameters():
                                    compression_param_ids.add(id(param))
                                print(f"Found MLPCompression-like module '{attr_name}' at layer {layer_idx}")
                                break
            
            # Also collect other compression-specific component parameters
            compression_components = ['linear_compress', 'compress_layer', 'compression_layer']
            for component_name in compression_components:
                if hasattr(compressed_layer, component_name):
                    component = getattr(compressed_layer, component_name)
                    if hasattr(component, 'parameters'):
                        for param in component.parameters():
                            compression_param_ids.add(id(param))
                        print(f"Found {component_name} parameters at layer {layer_idx}")
                elif hasattr(compressed_layer, 'self_attn') and hasattr(compressed_layer.self_attn, component_name):
                    component = getattr(compressed_layer.self_attn, component_name)
                    if hasattr(component, 'parameters'):
                        for param in component.parameters():
                            compression_param_ids.add(id(param))
                        print(f"Found attention {component_name} parameters at layer {layer_idx}")
        
        # Now freeze only the parameters that are NOT compression-specific
        frozen_count = 0
        unfrozen_count = 0
        
        for param in self.model.parameters():
            if id(param) in compression_param_ids:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Froze {frozen_count} pretrained parameters, kept {unfrozen_count} compression parameters trainable at layers {self.compressed_layer_indices}")
        
        # Print trainable parameters summary to verify
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print the number of trainable parameters and percentage."""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percentage = 100 * trainable_params / all_param if all_param > 0 else 0
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {trainable_percentage:.2f}")

    # All the following methods are identical to LayerWrapperPretrained
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the wrapped model."""
        self.model.save_pretrained(save_directory, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        """Get the state dict of the wrapped model."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict into the wrapped model."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Move the model to device/dtype (skip for quantized models)."""
        if not self._is_quantized_model(self.model):
            self.model = self.model.to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Return named model parameters."""
        return self.model.named_parameters()
    
    @property
    def config(self):
        """Access to model config."""
        return self.model.config
    
    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device
    
    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.model.get_output_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        return self.model.set_input_embeddings(value)
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        return self.model.set_output_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        """Resize token embeddings."""
        return self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
    
    def tie_weights(self):
        """Tie input and output embeddings."""
        return self.model.tie_weights()
    
    def get_decoder(self):
        """Get the decoder part of the model."""
        return self.model.get_decoder()
    
    def get_encoder(self):
        """Get the encoder part of the model (if exists)."""
        if hasattr(self.model, 'get_encoder'):
            get_encoder_method = getattr(self.model, 'get_encoder')
            if callable(get_encoder_method):
                return get_encoder_method()
        return None
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing."""
        return self.model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        return self.model.gradient_checkpointing_disable() 