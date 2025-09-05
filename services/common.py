DEBUG_FLEX_ATTENTION = True  # Set to True to enable debug prints


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
    debug_print(
        f"  memory_format: {tensor.memory_format if hasattr(tensor, 'memory_format') else 'N/A'}"
    )


def check_system_state(torch):
    """Check system state that might affect FlexAttention compilation"""
    debug_print("System state check:")
    debug_print(f"  PyTorch version: {torch.__version__}")
    debug_print(f"  CUDA available: {torch.cuda.is_available()}")
    debug_print(
        f"  Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}"
    )
    debug_print(
        f"  Is compiled: {hasattr(torch, '_C') and hasattr(torch._C, '_get_current_static_runtime')}"
    )
    debug_print(f"  Gradient enabled: {torch.is_grad_enabled()}")
    debug_print(f"  In inference mode: {torch.is_inference_mode_enabled()}")
