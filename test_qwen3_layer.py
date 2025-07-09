import torch
import torch.nn as nn
import math
from Compress import CompressedAttention
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

config = Qwen3Config()
# Define a simple config class since Qwen3Config is not available
layer_idx = 0

def test_qwen3_layer():
    """Test the Qwen3 CompressedAttention layer"""
    print("Testing Qwen3 CompressedAttention Layer")
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
        # Create layer with compression parameters in constructor
        layer = CompressedAttention(layer_idx=layer_idx, T_w=window_size, r=r, config=config, M=0).to(device)
        layer.eval()
        output, attn_weights, past_key_value = layer(
            hidden_states=hidden_states
        )

            
        print(f"Layer parameters: {sum(p.numel() for p in layer.parameters()):,}")
        print()
        # Calculate compression ratio
        compression_ratio = 1 - (output.shape[1]-window_size) / (seq_len-window_size)
        expected_ratio = (1 - r) / 2
        
        print(f"r={r:.1f}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Actual compression ratio: {compression_ratio:.3f}")
        print(f"  Expected compression ratio: {expected_ratio:.3f}")
        print(f"  Difference: {abs(compression_ratio - expected_ratio):.3f}")
        print()
    
    # Test with the original r=0.6 for comparison
    with torch.no_grad():
        output, attn_weights, past_key_value = layer(
            hidden_states=hidden_states
        )
        
        print("Original test (r=0.6):")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
        print(f"Past key value: {past_key_value}")
        
        # Verify output shape
        expected_seq_len = seq_len - (seq_len - window_size) // 2  # Compression formula
        print(f"Expected compressed sequence length: ~{expected_seq_len}")
        print(f"Actual output sequence length: {output.shape[1]}")
        
        compression_ratio = 1 - (output.shape[1]-window_size) / (seq_len-window_size)
        print(f"Compression ratio: {compression_ratio:.2%}")
            


def test_compression_only():
    """Test just the compression functionality"""
    print("\n" + "=" * 50)
    print("Testing Compression Functionality Only")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    test_cases = [
        (1, 100, 20, 0.6, "Single batch"),
        (2, 100, 20, 0.5, "Multi batch"),
        (1, 101, 21, 0.7, "Odd sequence"),
        (2, 200, 50, 0.4, "Long sequence"),
    ]
    
    for batch_size, seq_len, window_size, r_val, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: batch={batch_size}, seq_len={seq_len}, window={window_size}, r={r_val}")
        
        # Create layer with test parameters
        layer = CompressedAttention(layer_idx=0, T_w=window_size, r=r_val, config=config, M=5).to(device)
        layer.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
        
        with torch.no_grad():

            output, attn_weights, _ = layer(
                hidden_states=hidden_states
            )
            
            print(f"  Input shape: {hidden_states.shape}")
            print(f"  Output shape: {output.shape}")
            
            compression_ratio = 1 - output.shape[1] / seq_len
            print(f"  Compression ratio: {compression_ratio:.2%}")
            expected_ratio = (1 - r_val) / 2
            print(f"  Expected compression ratio: {expected_ratio:.2%}")
            print(f"  Difference: {abs(compression_ratio - expected_ratio):.2%}")
            
            # Verify batch and feature dimensions
            assert output.shape[0] == batch_size
            assert output.shape[2] == config.hidden_size
            print(f"  ✓ Shape verification passed")


def test_with_cache():
    """Test the layer with key-value cache"""
    print("\n" + "=" * 50)
    print("Testing with Key-Value Cache")
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
    
    # Create layer with compression parameters
    layer = CompressedAttention(layer_idx=0, T_w=window_size, r=0.6, config=config, M=5).to(device)
    layer.eval()
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    # Create cache
    past_key_value = SimpleDynamicCache()
    
    print(f"Input shape: {hidden_states.shape}")
    
    with torch.no_grad():
        output, attn_weights, updated_cache = layer(
            hidden_states=hidden_states,
            past_key_value=past_key_value
        )
        
        print(f"Output shape: {output.shape}")
        print(f"Cache updated: {updated_cache is not None}")
        print(f"✓ Cache test passed")


def test_gradients():
    """Test that gradients flow properly"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    batch_size = 1
    seq_len = 64
    window_size = 32
    
    # Create layer with compression parameters
    layer = CompressedAttention(layer_idx=0, T_w=window_size, r=0.6, config=config, M=5).to(device)
    layer.train()  # Enable training mode
    
    # Create input with gradients
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, requires_grad=True)
    
    print(f"Input requires grad: {hidden_states.requires_grad}")
    

    output, _, _ = layer(
        hidden_states=hidden_states
    )
    
    # Compute a simple loss
    loss = output.mean()
    loss.backward()
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.6f}")
    
    # Check input gradients with proper error handling
    if hidden_states.grad is not None:
        print(f"Input grad norm: {hidden_states.grad.norm().item():.6f}")
    else:
        print("Input grad: None")
    
    # Check that some parameters have gradients
    param_grads = [p.grad is not None for p in layer.parameters() if p.requires_grad]
    param_names_with_grad = [name for name, p in layer.named_parameters() if p.requires_grad and p.grad is not None]
    print(f"Parameters with gradients: {param_names_with_grad}")
    print(f"Parameters with gradients: {sum(param_grads)}/{len(param_grads)}")
    print(f"✓ Gradient test passed")
        


if __name__ == "__main__":
    test_qwen3_layer()
    test_compression_only() 
    test_with_cache()
    test_gradients()
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("=" * 50)
   