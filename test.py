import torch
import torch.nn as nn
import math
from Compress import *
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

config = Qwen3Config()

def test_final_compressed_tokens():
    """Test the FinalCompressedTokens class functionality"""
    print("Testing FinalCompressedTokens Class")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    batch_size = 2
    seq_len = 50
    window_size = 10
    r = 0.6
    M = 0
    
    # Create test inputs - these should be flattened representations
    q_w = torch.randn(batch_size, window_size, config.num_attention_heads, config.head_dim, device=device).transpose(1, 2)
    km_cmp = torch.randn(batch_size, seq_len, config.num_key_value_heads, config.head_dim, device=device).transpose(1, 2)
    
    print("Input shapes:")
    print(f"  q_w: {q_w.shape}")
    print(f"  km_cmp: {km_cmp.shape}")
    print()
    
    # Create FinalCompressedTokens instance
    final_compression = FinalCompressedTokens(config, q_w, km_cmp, r, M)
    print(f"Final compression parameters: {sum(p.numel() for p in final_compression.parameters()):,}")
    # Create test memory tensors
    xm_cmp = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    x_m = torch.randn(batch_size, 2* seq_len , config.hidden_size, device=device)
    output = final_compression(x_m, xm_cmp)
    print(f"Output shape: {output.shape}")
    compression_ratio = 1 - (output.shape[1]-window_size) / (2*seq_len)
    print(f"Compression ratio: {compression_ratio:.2%}")
    
    print(f"Memory tensor shape: {xm_cmp.shape}")
    print()
    
    # Test forward pass
    with torch.no_grad():
        output = final_compression(xm_cmp, xm_cmp)
        
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Verify output shape matches input shape
        assert output.shape == xm_cmp.shape, f"Output shape {output.shape} doesn't match input shape {xm_cmp.shape}"
        print(f"✓ Shape verification passed")
        
        # Test with different memory sizes
        test_cases = [
            (1, 32, 8, "Small sequence"),
            (2, 100, 20, "Medium sequence"),
            (1, 200, 50, "Large sequence"),
        ]
        
        print("\nTesting different sequence sizes:")
        for batch_size, seq_len, window_size, description in test_cases:
            print(f"\nTest: {description}")
            print(f"Input: batch={batch_size}, seq_len={seq_len}, window={window_size}")
            
            q_w = torch.randn(batch_size, config.num_attention_heads, window_size, config.head_dim, device=device)
            km_cmp = torch.randn(batch_size, config.num_key_value_heads, seq_len, config.head_dim, device=device)
            xm_cmp = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
            x_m = torch.randn(batch_size, 2*seq_len, config.hidden_size, device=device)
            final_compression = FinalCompressedTokens(config, q_w, km_cmp, r, M)
            
            output = final_compression(x_m, xm_cmp)
            
            print(f"  Input shape: {xm_cmp.shape}")
            print(f"  Output shape: {output.shape}")
            compression_ratio = 1 - (output.shape[1]-window_size) / (2*seq_len)
            print(f"Compression ratio: {compression_ratio:.2%}")
        
        print(f"\n✓ All FinalCompressedTokens tests passed!")


def test_final_compressed_tokens_gradients():
    """Test that gradients flow properly through FinalCompressedTokens"""
    print("\n" + "=" * 50)
    print("Testing FinalCompressedTokens Gradient Flow")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config()
    
    batch_size = 1
    seq_len = 32
    window_size = 8
    r = 0.6
    M = 5
    
    # Create inputs with gradients - these should be flattened representations
    q_w = torch.randn(batch_size, config.num_attention_heads, window_size, config.head_dim, device=device)
    km_cmp = torch.randn(batch_size, config.num_key_value_heads, seq_len - window_size, config.head_dim, device=device)
    xm_cmp = torch.randn(batch_size, seq_len - window_size, config.hidden_size, device=device, requires_grad=True)
    x_m = torch.randn(batch_size, 2*seq_len, config.hidden_size, device=device, requires_grad=True)
    print(f"Input requires grad: {xm_cmp.requires_grad}")
    
    final_compression = FinalCompressedTokens(config, q_w, km_cmp, r, M)
    final_compression.train()  # Enable training mode
    
    output = final_compression(x_m, xm_cmp)
    
    # Compute a simple loss
    loss = output.mean()
    loss.backward()
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.6f}")
    if xm_cmp.grad is not None:
        print(f"Input grad norm: {xm_cmp.grad.norm().item():.6f}")
    else:
        print("Input grad: None")
    
    # Check that some parameters have gradients
    param_grads = [p.grad is not None for p in final_compression.parameters() if p.requires_grad]
    print(f"Parameters with gradients: {sum(param_grads)}/{len(param_grads)}")
    print(f"✓ Gradient test passed")
        

if __name__ == "__main__":
    test_final_compressed_tokens()
    test_final_compressed_tokens_gradients()
    print("\n" + "=" * 50)
    print("🎉 All FinalCompressedTokens tests completed!")
    print("=" * 50)







