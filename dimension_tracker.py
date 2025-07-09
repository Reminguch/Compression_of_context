import torch
from unsloth import FastLanguageModel

def track_layer_dimensions(model, tokenizer, text="Test input for dimension tracking"):
    """Track output dimensions through all layers during training-style forward pass"""
    layer_dims = {}
    
    def dimension_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                shape = output[0].shape
            else:
                shape = output.shape
            layer_dims[name] = shape
            print(f"{name}: {shape}")
        return hook
    
    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.model.layers):
        hook = layer.register_forward_hook(dimension_hook(f"layer_{i}"))
        hooks.append(hook)
    
    # Add hooks for final components
    if hasattr(model.model, 'norm'):
        hooks.append(model.model.model.norm.register_forward_hook(dimension_hook("norm")))
    if hasattr(model, 'lm_head'):
        hooks.append(model.model.lm_head.register_forward_hook(dimension_hook("lm_head")))
    
    # Training-style forward pass
    inputs = tokenizer(text, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Model device: {device}")
    print(f"Input device: {inputs['input_ids'].device}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Final output shape: {outputs.logits.shape}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return layer_dims

# Example text for reference (not executed)
sample_context = """
Supernatural Phenomena and Paranormal Investigations

The small town of Millbrook had been experiencing unexplained phenomena for over a century. Local historians documented strange occurrences dating back to the 1800s, including mysterious lights in the old cemetery, unexplained sounds from the abandoned mill, and reports of apparitions near the town's founding marker.

Recent investigations by paranormal researchers have revealed several key patterns:
1. Most supernatural activity occurs between midnight and 3 AM
2. The phenomena intensify during the new moon phases
3. Electromagnetic field disturbances are consistently detected in affected areas
4. Local residents report feelings of unease and sudden temperature drops

The town's most famous case involves the Miller family farmhouse, where witnesses have reported seeing a woman in white walking through the fields at night. This apparition, known locally as "The Weeping Lady," has been documented by multiple independent sources over the past 50 years.

Understanding these supernatural events requires careful analysis of both historical records and modern investigative techniques.
"""

sample_question = """Question: Based on the patterns observed in Millbrook's supernatural phenomena, what time period would be most likely for paranormal investigators to encounter "The Weeping Lady" apparition?

Please provide your answer."""