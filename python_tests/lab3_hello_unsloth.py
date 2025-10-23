#!/usr/bin/env python3
"""
Lab 3: Hello Unsloth - Load and Infer
Test script for loading and inferring with a small LLM
Note: On Mac M3, we'll use transformers directly since Unsloth may not support MPS
"""

import time
import sys

def main():
    print("=" * 60)
    print("Lab 3: Hello LLM - Load and Infer")
    print("=" * 60)
    
    # Import libraries
    print("\n1️⃣ Importing libraries...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers imported successfully")
    except ImportError as e:
        print(f"Error importing libraries: {e}")
        print("Please install: pip install torch transformers")
        return False
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n✓ Using CUDA GPU")
    else:
        device = "cpu"
        print(f"\n✓ Using CPU (this will be slower)")
    
    # Load a small Qwen model (Qwen2.5-1.5B - works well on Mac M3)
    print("\n2️⃣ Loading model and tokenizer...")
    print("Using Qwen2.5-1.5B-Instruct (Apache 2.0, optimized for limited resources)")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else "cpu",  # MPS needs special handling
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if available
        if device == "mps":
            model = model.to(device)
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"  Model parameters: ~1.1B")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nThis is expected if you don't have enough RAM.")
        print("Try closing other applications and running again.")
        return False
    
    # 3️⃣ Run inference and measure performance
    print("\n3️⃣ Running inference...")
    
    # Format prompt for Qwen chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the principle of superposition in quantum mechanics in simple terms."}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nPrompt: Explain the principle of superposition in quantum mechanics...")
    
    def generate_response(prompt: str, max_new_tokens: int = 150):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        elapsed = end_time - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Count tokens (approximate)
        num_tokens = len(outputs[0])
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else float('inf')
        
        return response, elapsed, tokens_per_sec, num_tokens
    
    try:
        response, elapsed_time, tps, num_tokens = generate_response(prompt_text)
        
        print(f"\n✓ Generation complete!")
        print(f"\nResponse:\n{response}")
        print(f"\nMetrics:")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
        print(f"  Total tokens: {num_tokens}")
        print(f"  Tokens per second: {tps:.2f}")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        return False
    
    # 4️⃣ Record memory usage
    print("\n4️⃣ Recording memory usage...")
    
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  CUDA memory allocated: {allocated:.2f} GB")
        print(f"  CUDA memory reserved: {reserved:.2f} GB")
    elif device == "mps":
        # MPS doesn't have memory tracking like CUDA
        print(f"  MPS memory tracking not available")
        print(f"  Model size: ~2.2 GB (fp16)")
    else:
        print(f"  Running on CPU")
        print(f"  Model size: ~4.4 GB (fp32)")
    
    # Reflection
    print("\n" + "=" * 60)
    print("Lab 3 Complete!")
    print("=" * 60)
    print("\nReflection:")
    print("- Save these metrics for comparison with optimized models")
    print(f"- Baseline inference: {tps:.2f} tokens/sec on {device}")
    print("- Future labs will apply optimization techniques")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

