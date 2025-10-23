#!/usr/bin/env python3
"""
Test script for OpenAI-compatible API (works with vLLM, MLX-LM, or llama.cpp)
"""

import requests
import json
import sys
import os
import argparse
from pathlib import Path

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


def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as error:
        print(f"❌ Connection failed: {error}")
        print(f"   Make sure the server is running at: {BASE_URL}")
        return False


def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing /v1/models endpoint...")
    try:
        headers = get_headers()
        response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint working")
            print(f"   Available models: {[m['id'] for m in data.get('data', [])]}")
            return True
        elif response.status_code == 401:
            print(f"❌ Authorization failed: {response.status_code}")
            print(f"   Set API_KEY environment variable or use --api-key flag")
            return False
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as error:
        print(f"❌ Request failed: {error}")
        return False


def test_chat_completion():
    """Test chat completion endpoint"""
    print("\n🔍 Testing /v1/chat/completions endpoint...")
    
    # Try to get model name from /v1/models endpoint first
    model_name = "qwen"  # default fallback
    try:
        headers = get_headers()
        models_response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=5)
        if models_response.status_code == 200:
            models = models_response.json().get('data', [])
            if models:
                model_name = models[0]['id']
    except:
        pass
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    print(f"   Sending request: {payload['messages'][0]['content']}")
    
    try:
        headers = get_headers()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✅ Chat completion successful!")
            print(f"   Response: {content}")
            
            usage = data.get('usage', {})
            if usage:
                print(f"   Tokens: {usage.get('total_tokens', 'N/A')} total")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as error:
        print(f"❌ Request failed: {error}")
        return False


def test_completion():
    """Test text completion endpoint"""
    print("\n🔍 Testing /v1/completions endpoint...")
    
    # Try to get model name from /v1/models endpoint first
    model_name = "qwen"  # default fallback
    try:
        headers = get_headers()
        models_response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=5)
        if models_response.status_code == 200:
            models = models_response.json().get('data', [])
            if models:
                model_name = models[0]['id']
    except:
        pass
    
    payload = {
        "model": model_name,
        "prompt": "The capital of France is",
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    print(f"   Sending prompt: {payload['prompt']}")
    
    try:
        headers = get_headers()
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            text = data['choices'][0]['text']
            print(f"✅ Text completion successful!")
            print(f"   Response: {payload['prompt']}{text}")
            return True
        else:
            print(f"❌ Text completion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as error:
        print(f"❌ Request failed: {error}")
        return False


def main():
    global BASE_URL, API_KEY
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test local server
  python test_api.py

  # Test RunPod deployment
  python test_api.py --url https://abc123-8000.proxy.runpod.net --api-key sk-yourkey

  # Use environment variables
  export API_BASE_URL=https://abc123-8000.proxy.runpod.net
  export API_KEY=sk-yourkey
  python test_api.py
        """
    )
    parser.add_argument(
        "--url",
        help="Base URL of the API (default: http://localhost:8000)",
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
    
    print("=" * 60)
    print("OpenAI-Compatible API Test Suite")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    if API_KEY:
        print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else ''}")
    else:
        print("API Key: None (testing without authentication)")
    print("=" * 60)
    
    # Run tests
    results = []
    results.append(("Models", test_models()))
    results.append(("Chat Completion", test_chat_completion()))
    results.append(("Text Completion", test_completion()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

