import time
import torch

from unsloth import FastLanguageModel
from Model_inference_customLoss import InferenceEngine
from decoder_wrapper import Qwen3Compressed


def benchmark_inference():
    """Benchmark inference time for original vs compressed model."""
    # Same parameters used elsewhere in the project
    MODEL_NAME = "unsloth/Qwen3-1.7B"
    T_W = 100
    R = 0.8
    M = 10
    COMPRESSED_LAYER_IDX = [10]

    # Load original model
    original_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Build compressed model wrapping the original (fresh copy)
    compressed_model = Qwen3Compressed(
        original_model=original_model,
        compressed_layer_indices=COMPRESSED_LAYER_IDX,
        T_w=T_W,
        r=R,
        M=M,
    )

    # Put both models in eval mode (no dropout) just like trainer forward pass
    original_model.eval()
    compressed_model.eval()

    # Single example from the supernatural dataset (same as used in training script)
    sample_context = (
        """
Supernatural Phenomena and Paranormal Investigations

The small town of Millbrook had been experiencing unexplained phenomena for over a century. \
Local historians documented strange occurrences dating back to the 1800s, including mysterious lights in the old cemetery, unexplained sounds from the abandoned mill, and reports of apparitions near the town's founding marker.

Recent investigations by paranormal researchers have revealed several key patterns:\n1. Most supernatural activity occurs between midnight and 3 AM\n2. The phenomena intensify during the new moon phases\n3. Electromagnetic field disturbances are consistently detected in affected areas\n4. Local residents report feelings of unease and sudden temperature drops

The town's most famous case involves the Miller family farmhouse, where witnesses have reported seeing a woman in white walking through the fields at night. This apparition, known locally as \"The Weeping Lady,\" has been documented by multiple independent sources over the past 50 years.

Understanding these supernatural events requires careful analysis of both historical records and modern investigative techniques.
"""
    )

    sample_question = (
        """Question: Based on the patterns observed in Millbrook's supernatural phenomena, what time period would be most likely for paranormal investigators to encounter \"The Weeping Lady\" apparition?

Please provide your answer."""
    )

    prompt = f"Context:\n{sample_context}\n\n{sample_question}"

    # Tokenize like the trainer (no special chat template)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to("cuda") for k, v in encoded.items()}

    # Training-like forward pass benchmark (no gradients)
    def timed_forward(model):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**encoded)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    original_time = timed_forward(original_model)
    compressed_time = timed_forward(compressed_model)

    print("Original model forward time:   {:.2f} seconds".format(original_time))
    print("Compressed model forward time: {:.2f} seconds".format(compressed_time))


if __name__ == "__main__":
    benchmark_inference() 