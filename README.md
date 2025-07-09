# Token Compression and Context Management

A PyTorch implementation of token compression and selective expansion techniques for efficient context management in transformer models.

## Features

### 🔄 Token Compression
- **TokenCompressor**: Compresses token pairs via MLP with learnable positional embeddings
- Handles odd-length sequences by preserving the last token
- Maintains window of recent tokens unchanged

### 🎯 Importance Scoring
- **WindowAttentionScore**: Computes attention-based importance scores
- Window tokens attend to memory tokens to determine relevance
- Efficient handling when window size >= sequence length (no computation needed)

### 📊 Token Selection
- **select_tokens**: Selects top tokens based on importance using formula: `r*(T-T_w) + M*(1-r)/2`
- Flexible ratio parameter `r` for balancing selection strategy
- Maintains original token ordering

### 🔀 Selective Expansion
- **interleave_selected_pure_tensor**: Expands selected compressed tokens back to original pairs
- Pure tensor operations - no loops for maximum efficiency
- Handles multiple batches with different selection patterns

## Key Components

### TokenCompressor
```python
compressor = TokenCompressor(dim=512, hidden_dim=2048)
compressed = compressor(x, T_w=10)  # Keep last 10 tokens unchanged
```

### WindowAttentionScore
```python
window_attn = WindowAttentionScore(dim=512, num_heads=8)
importance_scores = window_attn.ImportanceScore(x, T_w=10)
```

### Token Selection
```python
selected_tokens = select_tokens(x, importance_scores, T_w=10, r=0.7)
```

### Selective Expansion
```python
expanded = interleave_selected_pure_tensor(x, x_compressed, selected_indices)
```

## Implementation Highlights

- ✅ **Pure PyTorch**: All operations use vectorized tensor operations
- ✅ **No Loops**: Completely loop-free implementations for GPU efficiency
- ✅ **Batch Support**: Handles multiple batches with different patterns
- ✅ **Differentiable**: Full gradient flow for training
- ✅ **Memory Efficient**: Minimal temporary tensor allocation

## Files

### Core Components
- `Compress.py`: Main compression implementation with CompressedDecoderLayer and CompressedAttention
- `decoder_wrapper.py`: ModelWrapperPretrained for integrating compression into pretrained models


### Dataset and Training
- 'create_long_dataset': Creates either Super-NaturalInstructions dataset for training or Longbench v2
- 'create_dataset_default': used for debugging, creates dataset with short math questions
- `Model_inference_customLoss.py`: Custom loss functions and inference engine
- `Unsloth.py`: Main training script using Unsloth for efficient fine-tuning

### Testing
- `test.py`: Basic functionality tests
- `test_compressed_decoder_layer.py`: Comprehensive tests for compressed layers
- `test_qwen3_layer.py`: Qwen3-specific layer testing

## Usage Example

```python
import torch
from unsloth import FastLanguageModel
from decoder_wrapper import ModelWrapperPretrained

# Load pretrained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=2048,
    load_in_4bit=False,
)

# Create compressed model
compressed_model = ModelWrapperPretrained(
    original_model=model,
    layer_idx=[10],  # Compress layer 10
    T_w=100,         # Window size
    r=0.8,           # Compression ratio
    M=100            # Maximum tokens
)

# Train with custom windowed loss
from Model_inference_customLoss import WindowedLossSFTTrainer

trainer = WindowedLossSFTTrainer(
    model=compressed_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    T_w=100
)

trainer.train()
```

## Applications

- **Long Sequence Processing**: Efficient handling of very long contexts
- **Memory Management**: Reduce memory usage while preserving important information
- **Parameter-Efficient Fine-tuning**: Only train compression components while keeping pretrained weights frozen
- **Transformer Optimization**: Reduce computational overhead in attention mechanisms

## Technical Details

The implementation uses advanced PyTorch features:
- **Selective Layer Compression**: Only specified layers are replaced with compressed versions
- **Parameter-Efficient Training**: ~82% of parameters remain frozen, only compression components are trainable
- **Streaming Dataset Support**: Handle large datasets efficiently
- **Custom Loss Functions**: Windowed supervision for better compression learning
- **Unsloth Integration**: Fast and memory-efficient training with quantization support

All operations are designed to be:
- GPU-accelerated
- Memory efficient
- Differentiable for end-to-end training
- Compatible with existing transformer architectures
