from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from Create_long_dataset import DatasetCreator
# from Create_dataset_default import DatasetCreator
from Model_inference_customLoss import WindowedLossSFTTrainer, InferenceEngine, layer_parameters, save_compressed_model, load_compressed_model
from decoder_wrapper import Qwen3Compressed
from typing import Dict
import copy
from dimension_tracker import track_layer_dimensions

# CUDA_VISIBLE_DEVICES=0 /bin/python3.10 /home/ilya/context_compression/Unsloth.py
print(1)
T_w = 1000000
r = 0.8
M = 1000000
layer_idx = [20]

# =============================================================
# Experiment hyper-parameters (adjust here, no code edits below)
# =============================================================
SEQ_LEN            = 2048       # Maximum input sequence length during training
MICRO_BATCH_SIZE   = 4         # Per-device batch size
GRAD_ACC_STEPS     = 8         # Gradient accumulation → global batch 32
MAX_STEPS          = 12   # ≈300 M tokens / 15 M parameters guideline
LEARNING_RATE      = 3e-3      # Peak LR for new compression layer
WARMUP_RATIO       = 0.05      # 5 % warm-up

# To keep memory manageable you can lower this. 600 k examples at 512 tokens
# roughly matches the 300 M-token budget suggested by Chinchilla.
DATASET_MAX_EXAMPLES = 600

fourbit_models = [
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

Pretrained_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-1.7B",
    max_seq_length = SEQ_LEN,   # Tunable context length for training
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)



# Pretrained_model_Lora = FastLanguageModel.get_peft_model(
#     Pretrained_model,
#     r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32,  # Best to choose alpha = rank or rank*2
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,   # We support rank stabilized LoRA
#     loftq_config = None,  # And LoftQ
# )


# Create model with compression at layer 10
compressed_model = Qwen3Compressed(
    original_model=Pretrained_model,
    compressed_layer_indices=layer_idx,
    T_w=T_w,
    r=r,
    M=M,
)
# print("Compressed Model created")
# compressed_model = load_compressed_model("saved_models/Compressed_layer10")

# # The compressed_model already has LoRA applied internally in ModelWrapperLora
# # No need to apply LoRA again
# my_model = copy.deepcopy(compressed_model)
# # my_model.freeze_pretrained()

my_model = compressed_model

my_model.freeze_pretrained()
my_model.print_trainable_parameters()


layer_parameters(my_model, 20)
print("model_created")

# Track layer dimensions during training-style forward pass
print("\n=== Tracking Layer Dimensions ===")
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
layer_dims = track_layer_dimensions(my_model, tokenizer, sample_context)
print("=== Dimension Tracking Complete ===\n")

# # Create LongBench v2 streaming dataset with optimized parameters
# from Create_longbench_v2_dataset import LongBenchV2DatasetCreator


creator = DatasetCreator(tokenizer, dataset_name="supernatural", use_streaming=True, seed=42)
streaming_dataset = creator.create_dataset(max_examples=DATASET_MAX_EXAMPLES)
# creator = DatasetCreator(tokenizer, chat_percentage=0.2)
# streaming_dataset = creator.create_dataset(max_length=512)  # Reduce dataset size for faster computations

# Verify the streaming dataset is working
print("Streaming dataset created successfully!")
print(f"Dataset type: {type(streaming_dataset)}")

# Use with tokenizer in streaming mode
# streaming_dataset can now be passed to your tokenizer

# Use the streaming dataset for training
training_dataset = streaming_dataset

# Enhanced WindowedLossSFTTrainer with streaming optimizations
trainer = WindowedLossSFTTrainer(
    T_w=T_w,  # Custom supervision window
    model=my_model,
    tokenizer=tokenizer,
    train_dataset=training_dataset,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    seq_length=SEQ_LEN,
    per_device_batch_size=MICRO_BATCH_SIZE,
    grad_acc_steps=GRAD_ACC_STEPS,
    warmup_ratio=WARMUP_RATIO,
)

# Optional: Add data validation
print("Validating streaming dataset...")
try:
    # Take a few samples to verify the streaming dataset is working
    sample_count = 0
    for sample in training_dataset:
        if isinstance(sample, dict):
            print(f"Sample {sample_count + 1} keys: {sample.keys()}")
            print(f"Sample text length: {len(sample.get('text', ''))}")
        else:
            print(f"Sample {sample_count + 1}: {type(sample)} - {sample}")
        sample_count += 1
        if sample_count >= 3:  # Just check first 3 samples
            break
    print(f"✓ Streaming dataset validation successful! Checked {sample_count} samples.")
except Exception as e:
    print(f"⚠️  Streaming dataset validation failed: {e}")
    print("Training may still work, but consider checking dataset creation.")





# The streaming dataset setup and trainer configuration are now handled above


# trainer = SFTTrainer(
#     model = my_model,
#     tokenizer = tokenizer,
#     train_dataset = combined_dataset,
#     eval_dataset = None, # Can set up evaluation!
#     args = SFTConfig(
#         dataset_text_field = "text",
#         per_device_train_batch_size = 1,  # Reduce to 1 to avoid chunking issues
#         gradient_accumulation_steps = 8,   # Increase GA to maintain effective batch size
#         warmup_steps = 5,
#         # num_train_epochs = 1, # Set this for 1 full training run.
#         max_steps = 30,
#         learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         report_to = "none", # Use this for WandB etc
#         dataloader_drop_last = True,  # Drop incomplete batches
#         remove_unused_columns = False,  # Keep all columns for debugging
#         max_length = 512,  # Add explicit max_length
#         # Disable potentially problematic features
#         packing = False,  # Disable packing which can cause tensor shape issues
#         padding_free = False,  # Disable padding-free training
#         group_by_length = False,  # Disable grouping by length
#     ),
# )


import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

# Start training
trainer.train()

save_compressed_model(my_model, tokenizer, "saved_models/Compressed_layer10")

# Example usage
inference_engine = InferenceEngine(my_model, tokenizer)

# Example 1: LongBench v2 style question with context and multiple choice
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

Please provide your answer with reasoning."""

messages = [
    {"role": "user", "content": f"Context:\n{sample_context}\n\n{sample_question}"}
]

# print("=== Example 1: Simple Math Problem ===")

# # Test the original model without compression to isolate the issue
# print("\n=== Testing Original Model (No Compression) ===")
# original_inference_engine = InferenceEngine(Pretrained_model, tokenizer)
# original_result = original_inference_engine.generate_response(
#     messages=messages,
#     max_new_tokens=100,
#     temperature=0.7,
#     enable_thinking=False,
#     stream=False
# )
# print("Original model result:")
# print(original_result)
# print("=" * 50)

print("\n=== Testing Compressed Model ===")

# -------------------------------------------------
# Collect outputs for later logging to a file
# -------------------------------------------------
test_outputs = []  # Each item is (header, string)

# Test 1: Try without thinking mode completely with anti-cutoff parameters
print("Test 1: Generation without thinking mode...")
result1 = inference_engine.generate_response(
    messages=messages,
    max_new_tokens=200,  # Reasonable token limit
    temperature=0.8,     # Higher temperature for better generation
    top_p=0.95,          # Higher top_p for diversity
    top_k=50,
    enable_thinking=False,
    stream=False
)
test_outputs.append(("Test 1", result1))

print("Full generated text:")
print(result1)
print(f"Response length: {len(result1)} characters")
print(f"Token count: {len(inference_engine.tokenizer.encode(result1))} tokens")
print("=" * 50)

# Test 1b: Try with more tokens to see if it's a length limit
print("Test 1b: Same test but with more max_new_tokens...")
result1b = inference_engine.generate_response(
    messages=messages,
    max_new_tokens=500,  # Much higher token limit
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    enable_thinking=False,
    stream=False
)
test_outputs.append(("Test 1b", result1b))

print("Full generated text (more tokens):")
print(result1b)
print(f"Response length: {len(result1b)} characters")
print(f"Token count: {len(inference_engine.tokenizer.encode(result1b))} tokens")
print("=" * 50)

#

# Test 2: Try a simpler prompt format
print("\nTest 2: Simple direct prompt...")
simple_text = "Solve (x + 2)^2 = 0. Show your work step by step."
simple_inputs = inference_engine.tokenizer(simple_text, return_tensors="pt").to("cuda")
try:
    simple_output = inference_engine.model.generate(
        **simple_inputs,
        max_new_tokens=256,
        temperature=0.7,  # Higher temperature for better sampling
        do_sample=True,
        pad_token_id=inference_engine.tokenizer.eos_token_id,
        use_cache=False,  # Disable cache for compressed models
        repetition_penalty=1.2,  # Prevent repetition
        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        top_p=0.9,
        top_k=50,
        early_stopping=True,
        past_key_values=None  # Explicitly set to None to avoid unsloth issues
    )
    simple_result = inference_engine.tokenizer.decode(simple_output[0], skip_special_tokens=True)
    print("Simple prompt result:")
    print(simple_result)
except Exception as e:
    print(f"Direct model generation failed: {e}")
    print("Using InferenceEngine instead...")
    simple_messages = [{"role": "user", "content": simple_text}]
    simple_result = inference_engine.generate_response(
        messages=simple_messages,
        max_new_tokens=256,
        temperature=0.1,
        stream=False
    )
    print("Simple prompt result (via InferenceEngine):")
    print(simple_result)
test_outputs.append(("Test 2", simple_result))
print("=" * 50)

# Test 3: Manual chat format without thinking
print("\nTest 3: Manual chat format...")
manual_text = "user\nSolve (x + 2)^2 = 0.\nassistant\n"
manual_inputs = inference_engine.tokenizer(manual_text, return_tensors="pt").to("cuda")
try:
    manual_output = inference_engine.model.generate(
        **manual_inputs,
        max_new_tokens=256,
        temperature=0.7,  # Higher temperature for better sampling
        do_sample=True,
        pad_token_id=inference_engine.tokenizer.eos_token_id,
        use_cache=False,  # Disable cache for compressed models
        repetition_penalty=1.2,  # Prevent repetition
        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        top_p=0.9,
        top_k=50,
        early_stopping=True,
        past_key_values=None  # Explicitly set to None to avoid unsloth issues
    )
    manual_result = inference_engine.tokenizer.decode(manual_output[0], skip_special_tokens=True)
    print("Manual format result:")
    print(manual_result)
except Exception as e:
    print(f"Direct model generation failed: {e}")
    print("Using InferenceEngine instead...")
    manual_messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
    manual_result = inference_engine.generate_response(
        messages=manual_messages,
        max_new_tokens=256,
        temperature=0.1,
        stream=False
    )
    print("Manual format result (via InferenceEngine):")
    print(manual_result)
test_outputs.append(("Test 3", manual_result))
print("=" * 50)

# Test 4: Check what the chat template actually produces
print("\nTest 4: Debug chat template...")
template_result = inference_engine.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("Chat template output:")
print(repr(template_result))
test_outputs.append(("Test 4 (Chat Template)", template_result))
print("=" * 50)

# Test 5: Test streaming with thinking disabled (anti-cutoff parameters)
print("\nTest 5: Streaming with thinking disabled...")
result_stream = inference_engine.generate_response(
    messages=messages,
    max_new_tokens=200,
    temperature=0.8,     # Higher temperature
    top_p=0.95,          # Higher top_p
    top_k=50,
    enable_thinking=False,
    stream=True
)
print(f"\nStream result: {result_stream}")
test_outputs.append(("Test 5 (Stream)", str(result_stream)))
print("=" * 50)

# Test 6: Simple generation method with conservative parameters
print("\nTest 6: Simple generation method...")
simple_result = inference_engine.generate_response(
    messages=messages,
    max_new_tokens=128,
    temperature=0.1,
    enable_thinking=False,
    stream=False
)
test_outputs.append(("Test 6", simple_result))
print("Simple generation result:")
print(simple_result)
print("=" * 50)

"""
After all tests are done, write the collected outputs to a file for later inspection.
"""

# Write all collected test outputs to a dedicated log file
log_path = os.path.join(os.path.dirname(__file__), "test_outputs.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for header, content in test_outputs:
        f.write(f"######## {header} ########\n")
        f.write(str(content))
        f.write("\n\n")

print(f"Saved detailed test outputs to {log_path}")
