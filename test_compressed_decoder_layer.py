import torch
import torch.nn as nn
import math
from Compress import CompressedDecoderLayer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

config = Qwen3Config()
def test_compressed_decoder_layer():
    """Test the CompressedDecoderLayer basic functionality"""
    print("Testing CompressedDecoderLayer")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create config
    config = Qwen3Config()
    print(f"Config: hidden_size={config.hidden_size}, heads={config.num_attention_heads}")
    print(f"Device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    window_size = 32
    layer_idx = 0
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    print("Input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print()
    
    # Test multiple compression ratios
    r_values = [0.2, 0.4, 0.6, 0.8]
    
    print("Testing different compression ratios:")
    print("-" * 40)
    
    for r in r_values:
        # Create layer with compression parameters
        layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=r, M=0).to(device)
        layer.eval()
        
        with torch.no_grad():
            outputs = layer(hidden_states=hidden_states)
            
            # Unpack outputs
            if len(outputs) == 1:
                output = outputs[0]
                attn_weights = None
            else:
                output = outputs[0]
                attn_weights = outputs[1] if len(outputs) > 1 else None
        
        print(f"r={r:.1f}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights: {attn_weights.shape if attn_weights is not None else None}")
        print(f"  Layer parameters: {sum(p.numel() for p in layer.parameters()):,}")
        
        # Calculate compression ratio
        compression_ratio = 1 - output.shape[1] / seq_len
        expected_ratio = (1 - r) / 2
        
        print(f"  Actual compression ratio: {compression_ratio:.3f}")
        print(f"  Expected compression ratio: {expected_ratio:.3f}")
        print(f"  Difference: {abs(compression_ratio - expected_ratio):.3f}")
        print()


def test_decoder_layer_with_attention_mask():
    """Test the layer with attention mask"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer with Attention Mask")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    batch_size = 2
    seq_len = 64
    window_size = 32
    layer_idx = 0
    
    # Create layer
    layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=0.6, M=5).to(device)
    layer.eval()
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    # Create attention mask (causal mask)
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    attention_mask = attention_mask.masked_fill(attention_mask == 1, -float('inf'))
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    # Create position IDs
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    with torch.no_grad():
        outputs = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        output = outputs[0]
        print(f"Output shape: {output.shape}")
        print(f"✓ Attention mask test passed")


def test_decoder_layer_with_cache():
    """Test the layer with key-value cache"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer with Key-Value Cache")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    # Create a simple cache class for testing
    class SimpleDynamicCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    batch_size = 1
    seq_len = 64
    window_size = 32
    layer_idx = 0
    
    # Create layer
    layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=0.6, M=5).to(device)
    layer.eval()
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    # Create cache
    past_key_value = SimpleDynamicCache()
    
    # Create cache position
    cache_position = torch.arange(seq_len, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Cache position shape: {cache_position.shape}")
    
    with torch.no_grad():
        outputs = layer(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
            use_cache=True
        )
        
        output = outputs[0]
        print(f"Output shape: {output.shape}")
        print(f"Cache updated: {len(past_key_value.key_cache) > 0}")
        print(f"✓ Cache test passed")


def test_decoder_layer_gradients():
    """Test that gradients flow properly through the decoder layer"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer Gradient Flow")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    batch_size = 1
    seq_len = 64
    window_size = 32
    layer_idx = 0
    
    # Create layer
    layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=0.6, M=5).to(device)
    layer.train()  # Enable training mode
    
    # Create input with gradients
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, requires_grad=True)
    
    print(f"Input requires grad: {hidden_states.requires_grad}")
    
    outputs = layer(hidden_states=hidden_states)
    output = outputs[0]
    
    # Compute a simple loss
    loss = output.mean()
    loss.backward()
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.6f}")
    
    # Check input gradients
    if hidden_states.grad is not None:
        print(f"Input grad norm: {hidden_states.grad.norm().item():.6f}")
    else:
        print("Input grad: None")
    
    # Check that some parameters have gradients
    param_grads = [p.grad is not None for p in layer.parameters() if p.requires_grad]
    param_names_with_grad = [name for name, p in layer.named_parameters() if p.requires_grad and p.grad is not None]
    
    print(f"Parameters with gradients: {len(param_names_with_grad)}")
    print(f"Sample parameters with gradients: {param_names_with_grad[:5]}")  # Show first 5
    print(f"Gradient coverage: {sum(param_grads)}/{len(param_grads)}")
    print(f"✓ Gradient test passed")


def test_decoder_layer_output_attention():
    """Test the layer with output_attentions=True"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer with Output Attentions")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    batch_size = 2
    seq_len = 64
    window_size = 32
    layer_idx = 0
    
    # Create layer
    layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=0.6, M=5).to(device)
    layer.eval()
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    
    with torch.no_grad():
        outputs = layer(
            hidden_states=hidden_states,
            output_attentions=True
        )
        
        output = outputs[0]
        attn_weights = outputs[1] if len(outputs) > 1 else None
        
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
        print(f"Number of outputs: {len(outputs)}")
        print(f"✓ Output attentions test passed")


def test_decoder_layer_edge_cases():
    """Test edge cases for the decoder layer"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer Edge Cases")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    layer_idx = 0
    
    test_cases = [
        (1, 32, 32, 0.5, "Window equals sequence length"),
        (2, 33, 32, 0.6, "Odd sequence length"),
        (1, 31, 32, 0.4, "Sequence shorter than window"),
        (3, 200, 50, 0.8, "Long sequence"),
        (1, 64, 16, 0.1, "High compression ratio"),
        (2, 64, 16, 0.9, "Low compression ratio"),
    ]
    
    for batch_size, seq_len, window_size, r_val, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Parameters: batch={batch_size}, seq_len={seq_len}, window={window_size}, r={r_val}")
        
        # Create layer
        layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=r_val, M=5).to(device)
        layer.eval()
        
        # Create input
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
        
        try:
            with torch.no_grad():
                outputs = layer(hidden_states=hidden_states)
                output = outputs[0]
                
                print(f"  Input shape: {hidden_states.shape}")
                print(f"  Output shape: {output.shape}")
                
                # Verify shapes
                assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
                assert output.shape[2] == config.hidden_size, f"Hidden size mismatch: {output.shape[2]} != {config.hidden_size}"
                
                compression_ratio = 1 - output.shape[1] / seq_len
                print(f"  Compression ratio: {compression_ratio:.3f}")
                print(f"  ✓ {description} passed")
                
        except Exception as e:
            print(f"  ✗ {description} failed: {str(e)}")


def test_decoder_layer_memory_efficiency():
    """Test memory usage of the decoder layer"""
    print("\n" + "=" * 50)
    print("Testing CompressedDecoderLayer Memory Efficiency")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    config = Qwen3Config()
    layer_idx = 0
    
    # Test with larger sequences to see memory impact
    batch_size = 1
    seq_len = 512
    window_size = 128
    
    print(f"Testing with large sequence: batch={batch_size}, seq_len={seq_len}")
    
    # Test different compression ratios
    for r in [0.2, 0.6, 0.9]:
        torch.cuda.empty_cache()  # Clear GPU memory
        
        # Measure initial memory
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Create layer and input
        layer = CompressedDecoderLayer(config=config, layer_idx=layer_idx, T_w=window_size, r=r, M=10).to(device)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
        
        setup_memory = torch.cuda.memory_allocated(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = layer(hidden_states=hidden_states)
            output = outputs[0]
        
        forward_memory = torch.cuda.memory_allocated(device)
        
        print(f"r={r:.1f}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Setup memory: {(setup_memory - initial_memory) / 1024**2:.2f} MB")
        print(f"  Forward memory: {(forward_memory - setup_memory) / 1024**2:.2f} MB")
        print(f"  Total memory: {(forward_memory - initial_memory) / 1024**2:.2f} MB")
        
        # Clean up
        del layer, hidden_states, outputs, output
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    test_compressed_decoder_layer()
    test_decoder_layer_with_attention_mask() 
    test_decoder_layer_with_cache()
    test_decoder_layer_gradients()
    test_decoder_layer_output_attention()
    test_decoder_layer_edge_cases()
    test_decoder_layer_memory_efficiency()
    
    print("\n" + "=" * 50)
    print("🎉 All CompressedDecoderLayer tests completed!")
    print("=" * 50) 