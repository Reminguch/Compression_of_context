#!/usr/bin/env python3
"""
Minimalistic test script to verify that the compressed model was saved successfully.
Tests loading the model and basic functionality.
"""

import os
import sys
import torch
import json
from pathlib import Path

def test_model_files_exist(model_path):
    """Test that all required model files exist."""
    print("🔍 Checking if model files exist...")
    
    required_files = [
        "compression_config.json",
        "model_info.json", 
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = Path(model_path) / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"  ✅ {file} exists ({file_path.stat().st_size} bytes)")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    # Check for model weights
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        print("  ❌ No .safetensors model weight files found")
        return False
    
    for file in safetensors_files:
        print(f"  ✅ {file.name} exists ({file.stat().st_size / (1024**3):.1f} GB)")
    
    print("  ✅ All required files exist!")
    return True

def test_compression_config(model_path):
    """Test that compression config is valid."""
    print("\n📋 Checking compression configuration...")
    
    config_path = Path(model_path) / "compression_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_keys = ['compressed_layer_indices', 'T_w', 'r']
    for key in required_keys:
        if key not in config:
            print(f"  ❌ Missing required config key: {key}")
            return False
        print(f"  ✅ {key}: {config[key]}")
    
    if 'M' in config:
        print(f"  ✅ M: {config['M']}")
    
    print("  ✅ Compression config is valid!")
    return True

def test_model_loading(model_path):
    """Test loading the compressed model."""
    print("\n🚀 Testing model loading...")
    
    try:
        # Import the loading function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from Model_inference_customLoss import load_compressed_model
        
        # Load the model (using CPU to avoid GPU memory issues in testing)
        print("  Loading model (this may take a moment)...")
        model, tokenizer = load_compressed_model(
            model_path, 
            max_seq_length=2048, 
            load_in_4bit=False, 
            device="cpu"  # Use CPU for testing to be safe
        )
        
        print(f"  ✅ Model loaded successfully!")
        print(f"  ✅ Model type: {type(model).__name__}")
        print(f"  ✅ Tokenizer type: {type(tokenizer).__name__}")
        
        # Check if it's the compressed model
        if hasattr(model, 'get_compression_params'):
            compression_params = model.get_compression_params()
            print(f"  ✅ Compression parameters: {compression_params}")
        else:
            print("  ⚠️  Model doesn't have compression parameters (might be loaded as regular model)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None, None

def test_basic_functionality(model, tokenizer):
    """Test basic model functionality."""
    print("\n🧪 Testing basic functionality...")
    
    if model is None or tokenizer is None:
        print("  ❌ Cannot test functionality - model not loaded")
        return False
    
    try:
        # Test tokenization
        test_text = "Hello, this is a test sentence."
        print(f"  Testing tokenization with: '{test_text}'")
        
        tokens = tokenizer.encode(test_text, return_tensors="pt")
        print(f"  ✅ Tokenization successful! Token shape: {tokens.shape}")
        
        # Test model forward pass
        print("  Testing model forward pass...")
        model.eval()
        
        print(f"  Debug: Model device = {model.device if hasattr(model, 'device') else 'unknown'}")
        print(f"  Debug: Tokens device = {tokens.device}")
        print(f"  Debug: Tokens dtype = {tokens.dtype}")
        
        with torch.no_grad():
            # Move tokens to same device as model if needed
            if hasattr(model, 'device'):
                print(f"  Moving tokens to model device: {model.device}")
                tokens = tokens.to(model.device)
                print(f"  Debug: Tokens after move - device = {tokens.device}")
            
            print("  Calling model forward pass...")
            try:
                outputs = model(tokens)
                print(f"  Debug: Forward pass completed, output type = {type(outputs)}")
            except Exception as forward_error:
                print(f"  ❌ Forward pass failed with error: {forward_error}")
                import traceback
                print(f"  Full traceback:")
                traceback.print_exc()
                raise forward_error
            
        print(f"  ✅ Forward pass successful!")
        if hasattr(outputs, 'logits'):
            print(f"  ✅ Output logits shape: {outputs.logits.shape}")
        elif hasattr(outputs, 'prediction_scores'):
            print(f"  ✅ Output prediction scores shape: {outputs.prediction_scores.shape}")
        else:
            print(f"  ✅ Output type: {type(outputs)}")
            if isinstance(outputs, tuple):
                print(f"  Debug: Tuple length = {len(outputs)}")
                for i, item in enumerate(outputs):
                    print(f"  Debug: Tuple[{i}] type = {type(item)}")
        
        # Test simple generation (just a few tokens to verify)
        print("  Testing simple generation...")
        try:
            generated = model.generate(
                tokens, 
                max_new_tokens=5, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as gen_error:
            print(f"  ❌ Generation failed: {gen_error}")
            return False
        print(f"  ✅ Generation successful!")
        print(f"  Generated: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔬 Testing Compressed Model Save/Load")
    print("=" * 50)
    
    # Model path
    model_path = "saved_models/Compressed_layer10"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path '{model_path}' does not exist!")
        return False
    
    print(f"📁 Testing model at: {model_path}")
    
    # Run tests
    all_passed = True
    
    # Test 1: File existence
    if not test_model_files_exist(model_path):
        all_passed = False
    
    # Test 2: Compression config
    if not test_compression_config(model_path):
        all_passed = False
    
    # Test 3: Model loading
    model, tokenizer = test_model_loading(model_path)
    if model is None:
        all_passed = False
    
    # Test 4: Basic functionality
    if model is not None:
        if not test_basic_functionality(model, tokenizer):
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Model saved successfully!")
    else:
        print("❌ SOME TESTS FAILED! Check the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)