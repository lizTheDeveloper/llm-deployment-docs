#!/usr/bin/env python3
"""
Test script for OpenAI-compatible API (works with vLLM, MLX-LM, or llama.cpp)
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


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
        print("   Make sure the server is running on port 8000")
        return False


def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing /v1/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint working")
            print(f"   Available models: {[m['id'] for m in data.get('data', [])]}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as error:
        print(f"❌ Request failed: {error}")
        return False


def test_chat_completion():
    """Test chat completion endpoint"""
    print("\n🔍 Testing /v1/chat/completions endpoint...")
    
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    print(f"   Sending request: {payload['messages'][0]['content']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
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
    
    payload = {
        "model": "qwen",
        "prompt": "The capital of France is",
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    print(f"   Sending prompt: {payload['prompt']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
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
    print("=" * 60)
    print("OpenAI-Compatible API Test Suite")
    print("=" * 60)
    
    # Run tests
    results = []
    results.append(("Health", test_health()))
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

