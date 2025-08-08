import torch
import torch.nn.functional as F
from Compress import CompressedAttention, MLPCompression, FinalCompressedTokens, Qwen3RMSNorm
from trl import SFTTrainer, SFTConfig

### Custom Loss function for training ###

class WindowedLossSFTTrainer(SFTTrainer):
    def __init__(
        self,
        T_w,
        model,
        tokenizer,
        train_dataset,
        max_steps: int = 12_000,
        learning_rate: float = 3e-3,
        seq_length: int = 512,
        per_device_batch_size: int = 4,
        grad_acc_steps: int = 8,
        warmup_ratio: float = 0.05,
        lr_scheduler_type: str = "cosine",
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        **kwargs,
    ):
        """
        Custom SFT trainer that supervises only the last `T_w` tokens **and** exposes
        the main hyper-parameters so they can be tuned from the experiment script.

        Args:
            T_w: supervision window size.
            model / tokenizer / train_dataset: standard TRL objects.
            max_steps: total optimisation steps.
            learning_rate: peak learning rate.
            seq_length: max sequence length used during training.
            per_device_batch_size: micro-batch size.
            grad_acc_steps: gradient accumulation steps.
            warmup_ratio: fraction of steps used for LR warm-up.
            lr_scheduler_type: scheduler type (e.g. "cosine").
            weight_decay: weight decay coefficient.
            logging_steps: how often to log.
        """

        self.T_w = T_w

        warmup_steps = int(max_steps * warmup_ratio)

        # Build the training arguments dynamically so they reflect our experiment setup
        training_args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            optim="adamw_8bit",
            lr_scheduler_type=lr_scheduler_type,
            bf16=True,
            weight_decay=weight_decay,
            seed=3407,
            report_to="none",
            max_length=seq_length,
            # Performance tweaks
            dataloader_pin_memory=True,
            dataloader_num_workers=1,
            # Force single GPU training to avoid NCCL issues
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False,
            # Explicitly disable distributed training
            local_rank=-1,
        )

        # Initialise the parent class with the newly built arguments
        super().__init__(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom compute_loss that supervises only the last T_w tokens
        
        Args:
            model: The model to compute loss for
            inputs: Input batch containing input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs along with loss
            **kwargs: Additional arguments that might be passed by parent class
        """
        # # Ensure the model is in training mode
        # model.train()
        
        # Ensure inputs are properly formatted tensors
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary")
            
        required_keys = ["input_ids", "attention_mask", "labels"]
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key in inputs: {key}")
            if not isinstance(inputs[key], torch.Tensor):
                inputs[key] = torch.tensor(inputs[key], device=model.device)
            if inputs[key].dim() == 1:
                inputs[key] = inputs[key].unsqueeze(0)
                
        # Fix attention mask dtype
        if inputs["attention_mask"].dtype != torch.bool:
            inputs["attention_mask"] = inputs["attention_mask"].bool()
            
        # Remove labels from inputs for forward pass
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        # Forward pass
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        
        if logits is None:
            raise ValueError("Model outputs must contain logits")
            
        # Ensure logits and labels have proper shape
        labels = inputs["labels"]
        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits tensor, got shape {logits.shape}")
            
        # Shift for causal prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Apply windowing if needed
        if shift_logits.size(1) > self.T_w:
            shift_logits = shift_logits[:, -self.T_w:, :].contiguous()
            shift_labels = shift_labels[:, -self.T_w:].contiguous()
            
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        
        return (loss, outputs) if return_outputs else loss



class CustomSFTTrainer(SFTTrainer):
    def __init__(self, model, tokenizer, train_dataset, **kwargs):
        # Create training arguments
        training_args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            # num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=30,
            learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use this for WandB etc
        )
        
        # Initialize parent SFTTrainer with correct parameter name
        super().__init__(
            model=model,
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
            train_dataset=train_dataset,
            eval_dataset=None,  # Can set up evaluation!
            args=training_args,
            **kwargs
        )

    def train(self):
        """Override train method to add custom training logic"""
        # Call parent train method
        return super().train()



class InferenceEngine:
    """Inference engine for model generation with configurable parameters"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_response(self, messages, max_new_tokens=256, temperature=0.7, 
                         top_p=0.8, top_k=20, enable_thinking=False, stream=True):
        """
        Generate response from messages using the model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            enable_thinking: Whether to enable thinking mode
            stream: Whether to stream the output
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Prepare generation kwargs with compatibility fixes for compressed models
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "use_cache": False,  # Disable KV cache to avoid past_key_values issues
            "do_sample": True,   # Enable sampling
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Prevent repetition
        }
        
        # Generate response without TextStreamer to avoid None return issues
        output_tokens = self.model.generate(**generation_kwargs)
        
        # Handle streaming by decoding and printing the result
        if stream:
            # Decode only the new tokens (skip the input prompt)
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_tokens[:, input_length:]
            generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            print(generated_text, flush=True)
            return "Generation completed (streamed to console)"
        else:
            # Decode the full output, but only return the new tokens after the input
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_tokens[:, input_length:]
            return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    
    def test_long_context_understanding(self, context, question, max_new_tokens=512, temperature=0.7):
        """
        Test the model's ability to understand and reason over long contexts
        
        Args:
            context: Long context text (document, article, code, etc.)
            question: Question about the context
            max_new_tokens: Maximum tokens for response
            temperature: Generation temperature
        """
        messages = [
            {
                "role": "user", 
                "content": f"""Please read the following context carefully and answer the question.

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context."""
            }
        ]
        
        return self.generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True
        )
    
    def test_multi_document_reasoning(self, documents, question, max_new_tokens=512, temperature=0.7):
        """
        Test reasoning across multiple documents
        
        Args:
            documents: List of document texts
            question: Question requiring reasoning across documents
            max_new_tokens: Maximum tokens for response
            temperature: Generation temperature
        """
        context = ""
        for i, doc in enumerate(documents, 1):
            context += f"\n\nDocument {i}:\n{doc}"
        
        return self.test_long_context_understanding(context, question, max_new_tokens, temperature)
    
    def test_document_analysis(self, document, analysis_type="summary", max_new_tokens=512, temperature=0.7):
        """
        Test document analysis capabilities
        
        Args:
            document: Document text to analyze
            analysis_type: Type of analysis ("summary", "key_points", "themes", "arguments")
            max_new_tokens: Maximum tokens for response
            temperature: Generation temperature
        """
        analysis_prompts = {
            "summary": "Please provide a comprehensive summary of this document.",
            "key_points": "What are the main key points discussed in this document?",
            "themes": "What are the central themes and topics covered in this document?",
            "arguments": "What are the main arguments presented in this document?",
            "timeline": "Extract and organize any temporal information or timeline from this document.",
            "relationships": "Identify and explain the relationships between different concepts, people, or entities mentioned in this document."
        }
        
        question = analysis_prompts.get(analysis_type, analysis_prompts["summary"])
        
        return self.test_long_context_understanding(document, question, max_new_tokens, temperature)


def layer_parameters(model, layer_idx):
    """
    Analyze parameters for a specific layer in the model.
    
    Args:
        model: The model to analyze
        layer_idx: The layer index to analyze (int)
    
    Returns:
        dict: Dictionary containing analysis results
    """
    layer_prefix = f".{layer_idx}."
    
    trainable_total = 0
    total_total = 0
    trainable_weights = 0
    frozen_weights = 0
    
    print(f"Parameters for layer {layer_idx}:\n")
    
    for name, param in model.named_parameters():
        if layer_prefix in name:
            count = param.numel()
            if param.requires_grad:
                trainable_total += count
                trainable_weights += 1
                print(f"[Trainable] {name}: {count:,}")
            else:
                frozen_weights += 1
                print(f"[Frozen   ] {name}: {count:,}")
            total_total += count
    
    print("\nSummary:")
    print(f"Total parameters in layer {layer_idx}:     {total_total:,}")
    print(f"Trainable parameters in layer {layer_idx}: {trainable_total:,}")
    print(f"Trainable weights in layer {layer_idx}: {trainable_weights:,}")
    print(f"Frozen weights in layer {layer_idx}: {frozen_weights:,}")

    if total_total > 0:
        trainable_ratio = trainable_total / total_total
        print(f"Trainable ratio: {trainable_ratio:.2%}")
    else:
        trainable_ratio = 0.0
        print("Trainable ratio: N/A (no parameters found)")
    
    return {
        'total_parameters': total_total,
        'trainable_parameters': trainable_total,
        'trainable_ratio': trainable_ratio,
        'layer_idx': layer_idx
    }

def save_compressed_model(my_model, tokenizer, path):
    """
    Save a compressed model (Qwen3Compressed) with its compression configuration.
    
    Args:
        my_model: The Qwen3Compressed model instance to save
        tokenizer: The tokenizer to save
        path: Path to save the model and tokenizer
    """
    import os
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save the tokenizer
    tokenizer.save_pretrained(path)
    
    # Check if this is a Qwen3Compressed model
    if hasattr(my_model, 'get_compression_params'):
        # Save the underlying base model, not the wrapper
        if hasattr(my_model, 'model'):
            my_model.model.save_pretrained(path)
            print("Saved underlying base model")
        else:
            # Fallback if model structure is different
            my_model.save_pretrained(path)
            print("Saved full model (no .model attribute found)")
        
        # Save compression configuration
        compression_config = my_model.get_compression_params()
        compression_config_path = os.path.join(path, "compression_config.json")
        
        with open(compression_config_path, 'w') as f:
            json.dump(compression_config, f, indent=2)
        
        print(f"Saved compressed model with compression config: {compression_config}")
        
        # Save model type info for loading
        model_info = {
            "model_type": "Qwen3Compressed",
            "base_model_type": type(my_model.model).__name__ if hasattr(my_model, 'model') else type(my_model).__name__
        }
        model_info_path = os.path.join(path, "model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    else:
        # Regular model saving
        my_model.save_pretrained(path)
        print("Saved regular model (no compression config)")
    
    print(f"Model and tokenizer saved to {path}")

def load_compressed_model(path, max_seq_length=2048, load_in_4bit=False, device="cuda"):
    """
    Load a compressed model (Qwen3Compressed) with its compression configuration.
    
    Args:
        path: Path where the model and tokenizer are saved
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit quantization
        device: Device to load the model on
    
    Returns:
        tuple: (model, tokenizer) where model is Qwen3Compressed if compression config exists
    """
    import os
    import json
    import torch
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Check if this is a compressed model
    compression_config_path = os.path.join(path, "compression_config.json")
    model_info_path = os.path.join(path, "model_info.json")
    
    if os.path.exists(compression_config_path) and os.path.exists(model_info_path):
        # Load compression configuration
        with open(compression_config_path, 'r') as f:
            compression_config = json.load(f)
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        print(f"Loading compressed model with config: {compression_config}")
        
        # Validate required config keys
        required_keys = ['compressed_layer_indices', 'T_w', 'r']
        for key in required_keys:
            if key not in compression_config:
                raise ValueError(f"Missing required compression config key: {key}")
        
        # Load the base model first
        try:
            if load_in_4bit:
                try:
                    from unsloth import FastLanguageModel
                    base_model, _ = FastLanguageModel.from_pretrained(
                        model_name=path,
                        max_seq_length=max_seq_length,
                        load_in_4bit=True,
                    )
                    print("Loaded base model using FastLanguageModel")
                except ImportError:
                    print("unsloth not available, falling back to transformers")
                    from transformers import AutoModelForCausalLM
                    from transformers.utils.quantization_config import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    base_model = AutoModelForCausalLM.from_pretrained(
                        path,
                        quantization_config=quantization_config,
                        device_map={"": device},  # Put entire model on specified device
                        torch_dtype=torch.float16,
                    )
            else:
                from transformers import AutoModelForCausalLM
                base_model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    device_map={"": device}  # Put entire model on specified device
                )
            
            print(f"Successfully loaded base model of type: {type(base_model)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load base model from {path}: {e}")
        
        # Import the Qwen3Compressed class
        try:
            from .decoder_wrapper import Qwen3Compressed
        except ImportError:
            try:
                from decoder_wrapper import Qwen3Compressed
            except ImportError:
                raise ImportError("Could not import Qwen3Compressed. Make sure decoder_wrapper.py is in the same directory or in the Python path.")
        
        # Create the compressed model wrapper
        try:
            compressed_model = Qwen3Compressed(
                original_model=base_model,
                compressed_layer_indices=compression_config['compressed_layer_indices'],
                T_w=compression_config['T_w'],
                r=compression_config['r'],
                M=compression_config.get('M', None)
            )
            
            print(f"Successfully created Qwen3Compressed model")
            print(f"Compression parameters: {compressed_model.get_compression_params()}")
            return compressed_model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Qwen3Compressed wrapper: {e}")
        
    else:
        # Load as regular model
        print("No compression config found, loading as regular model")
        
        if load_in_4bit:
            try:
                from unsloth import FastLanguageModel
                model, _ = FastLanguageModel.from_pretrained(
                    model_name=path,
                    max_seq_length=max_seq_length,
                    load_in_4bit=True,
                )
                return model, tokenizer
            except ImportError:
                print("unsloth not available, falling back to transformers")
                from transformers import AutoModelForCausalLM
                from transformers.utils.quantization_config import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    quantization_config=quantization_config,
                    device_map={"": device},  # Put entire model on specified device
                    torch_dtype=torch.float16,
                )
                return model, tokenizer
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map={"": device}  # Put entire model on specified device
            )
            return model, tokenizer