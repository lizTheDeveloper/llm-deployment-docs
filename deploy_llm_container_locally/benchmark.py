#!/usr/bin/env python3
"""
Benchmark script to measure tokens/second performance
"""

import requests
import time
import json
import sys
import os
import argparse
from pathlib import Path
from statistics import mean, stdev

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Get configuration from environment variables
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", None)

def get_headers():
    """Get headers with optional authorization"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def get_model_name():
    """Get the actual model name from the API"""
    try:
        headers = get_headers()
        response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get('data', [])
            if models:
                return models[0]['id']
    except:
        pass
    return "qwen"  # fallback


def benchmark_chat(num_requests=20, max_tokens=50):
    """Run benchmark on chat completions endpoint"""
    
    # Get the actual model name
    model_name = get_model_name()
    
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
    print(f"Model: {model_name}")
    print(f"Requests: {num_requests}")
    print(f"Max tokens per request: {max_tokens}")
    if API_KEY:
        print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"{'='*60}\n")
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        
        try:
            headers = get_headers()
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark local server
  python benchmark.py 20 50

  # Benchmark RunPod deployment
  python benchmark.py --url https://abc123-8000.proxy.runpod.net --api-key sk-yourkey 20 50

  # Use environment variables
  export API_BASE_URL=https://abc123-8000.proxy.runpod.net
  export API_KEY=sk-yourkey
  python benchmark.py 20 50
        """
    )
    parser.add_argument(
        "num_requests",
        nargs="?",
        type=int,
        default=20,
        help="Number of requests to send (default: 20)"
    )
    parser.add_argument(
        "max_tokens",
        nargs="?",
        type=int,
        default=50,
        help="Maximum tokens per request (default: 50)"
    )
    parser.add_argument(
        "--url",
        help="Base URL of the API",
        default=BASE_URL
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (Bearer token)",
        default=API_KEY
    )
    
    args = parser.parse_args()
    
    # Update globals from arguments
    BASE_URL = args.url
    API_KEY = args.api_key
    
    # Check if server is accessible
    print(f"Checking server at {BASE_URL}...")
    try:
        headers = get_headers()
        response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"✅ Server is accessible\n")
        elif response.status_code == 401:
            print(f"❌ Authorization failed. Check your API key.")
            sys.exit(1)
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            print(f"   Continuing anyway...\n")
    except requests.exceptions.RequestException as error:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"   Error: {error}")
        print(f"   Make sure the server is running and accessible")
        sys.exit(1)
    
    # Run benchmark
    benchmark_chat(num_requests=args.num_requests, max_tokens=args.max_tokens)

