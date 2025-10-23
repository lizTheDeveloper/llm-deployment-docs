#!/usr/bin/env python3
"""
Benchmark script to measure tokens/second performance
"""

import requests
import time
import json
from statistics import mean, stdev

BASE_URL = "http://localhost:8000"

def benchmark_chat(num_requests=20, max_tokens=50):
    """Run benchmark on chat completions endpoint"""
    
    prompts = [
        "Explain what Docker is in one sentence.",
        "What is machine learning?",
        "Define artificial intelligence briefly.",
        "What is Python used for?",
        "Explain cloud computing simply.",
        "What is a neural network?",
        "Define DevOps in simple terms.",
        "What is Kubernetes?",
        "Explain what an API is.",
        "What is a container?",
        "Define microservices architecture.",
        "What is continuous integration?",
        "Explain version control briefly.",
        "What is a database?",
        "Define REST API simply.",
        "What is Git used for?",
        "Explain what CI/CD means.",
        "What is serverless computing?",
        "Define infrastructure as code.",
        "What is a virtual machine?",
    ]
    
    print(f"🚀 Starting Benchmark")
    print(f"{'='*60}")
    print(f"Endpoint: {BASE_URL}/v1/chat/completions")
    print(f"Requests: {num_requests}")
    print(f"Max tokens per request: {max_tokens}")
    print(f"{'='*60}\n")
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                tokens = data['usage']['completion_tokens']
                tok_per_sec = tokens / elapsed if elapsed > 0 else 0
                
                total_tokens += tokens
                total_time += elapsed
                
                results.append({
                    'request': i + 1,
                    'tokens': tokens,
                    'time': elapsed,
                    'tok_per_sec': tok_per_sec
                })
                
                # Progress indicator
                bar_width = 30
                progress = (i + 1) / num_requests
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"\r[{bar}] {i+1}/{num_requests} | Last: {tok_per_sec:.1f} tok/s", end='', flush=True)
                
            else:
                print(f"\n❌ Request {i+1} failed: {response.status_code}")
                
        except Exception as error:
            print(f"\n❌ Request {i+1} error: {error}")
    
    print("\n")
    
    # Calculate statistics
    if results:
        tok_per_sec_values = [r['tok_per_sec'] for r in results]
        avg_tok_per_sec = mean(tok_per_sec_values)
        std_tok_per_sec = stdev(tok_per_sec_values) if len(tok_per_sec_values) > 1 else 0
        min_tok_per_sec = min(tok_per_sec_values)
        max_tok_per_sec = max(tok_per_sec_values)
        
        overall_tok_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 Benchmark Results")
        print(f"{'='*60}")
        print(f"Total requests: {len(results)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\n🎯 Overall Performance:")
        print(f"  Average: {overall_tok_per_sec:.1f} tokens/second")
        print(f"\n📈 Per-Request Statistics:")
        print(f"  Mean:   {avg_tok_per_sec:.1f} tok/s")
        print(f"  Std:    {std_tok_per_sec:.1f} tok/s")
        print(f"  Min:    {min_tok_per_sec:.1f} tok/s")
        print(f"  Max:    {max_tok_per_sec:.1f} tok/s")
        print(f"\n⏱️  Timing:")
        print(f"  Avg time/request: {total_time/len(results):.2f}s")
        print(f"  Avg tokens/request: {total_tokens/len(results):.1f}")
        print(f"{'='*60}\n")
        
        # Show some example responses
        print("💬 Sample Responses:")
        for i in range(min(3, len(results))):
            print(f"\n  Request {i+1}: {results[i]['tokens']} tokens in {results[i]['time']:.2f}s ({results[i]['tok_per_sec']:.1f} tok/s)")
    else:
        print("❌ No successful requests completed")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print(f"❌ Server not healthy at {BASE_URL}")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("   Make sure the MLX server is running:")
        print("   ./run_mlx_native.sh")
        sys.exit(1)
    
    # Run benchmark
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    benchmark_chat(num_requests=num_requests, max_tokens=max_tokens)

