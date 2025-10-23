# RunPod vLLM Deployment Guide

## Your Current Issue: 401 Unauthorized

The error shows vLLM is running but requires authentication. Here's how to fix it:

---

## Solution 1: Access with API Key (If Set)

If you started vLLM with `--api-key`, you need to pass it in requests:

```bash
# Get your RunPod proxy URL
POD_ID="your-pod-id"  # Find in RunPod console
PORT=8000             # Your vLLM port

# Test with API key
curl https://ija84s9k1w2wg9-8000.proxy.runpod.net/v1/models \
  -H "Authorization: Bearer sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
```

**Python example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url=f"https://{POD_ID}-{PORT}.proxy.runpod.net/v1",
    api_key="YOUR_API_KEY_HERE"  # The key you set when starting vLLM
)

response = client.chat.completions.create(
    model="unsloth/Qwen3-4B-unsloth-bnb-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

## Solution 2: Disable Authentication (Easiest for Demos)

Restart your vLLM container **without** the `--api-key` flag:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3-4B-unsloth-bnb-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
  # NO --api-key flag
```

Then access without authentication:
```bash
curl https://${POD_ID}-8000.proxy.runpod.net/v1/models
```

---

## How to Find Your RunPod Access URL

### Method 1: RunPod Console

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Find your pod
3. Look for "Connect" button or expand pod details
4. Your URL format: `https://[POD_ID]-[PORT].proxy.runpod.net`

### Method 2: From Pod ID

If your Pod ID is `abc123xyz` and vLLM runs on port `8000`:
```
https://abc123xyz-8000.proxy.runpod.net
```

---

## Exposing Ports in RunPod

### During Pod Creation:

1. Click "Deploy Pod"
2. Scroll to "Edit Template"
3. Find "Expose HTTP Ports" field
4. Add `8000` (or your vLLM port)
5. Deploy

### For Existing Pods:

1. Go to your pod
2. Click hamburger menu (bottom-left)
3. Select "Edit Pod"
4. Add `8000` to "Expose HTTP Ports"
5. Save

---

## Complete Setup Example

### 1. Your Dockerfile (already working)
```dockerfile
FROM vllm/vllm-openai:latest

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "unsloth/Qwen3-4B-unsloth-bnb-4bit", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--dtype", "float16", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.9"]
```

**Key points:**
- ✅ `--host 0.0.0.0` - Listens on all interfaces (required!)
- ✅ `--port 8000` - Your service port
- ✅ No `--api-key` - Disables authentication for easier testing

### 2. RunPod Configuration

**In RunPod console:**
- Template → Edit Template
- Expose HTTP Ports: `8000`
- GPU: Any with 24GB+ VRAM (A40, A100, etc.)

### 3. Access Your API

Once deployed, get your Pod ID from the console (looks like `abc123xyz456`):

```bash
# Test connection
curl https://abc123xyz456-8000.proxy.runpod.net/v1/models

# Test chat
curl https://abc123xyz456-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq
```

---

## Troubleshooting

### 401 Unauthorized
**Problem:** vLLM requires authentication
**Solutions:**
1. Remove `--api-key` flag from vLLM command
2. Or pass API key in Authorization header: `Authorization: Bearer YOUR_KEY`

### 404 Not Found on `/`
**Normal!** vLLM doesn't have a root endpoint. Try:
- `/v1/models`
- `/v1/chat/completions`
- `/health`

### 524 Timeout Error
**Problem:** RunPod HTTP proxy has 100-second timeout
**Solutions:**
- Reduce `--max-model-len` for faster responses
- Use smaller model for testing
- Implement streaming responses

### Can't Connect
**Checklist:**
- ✅ Port exposed in RunPod (8000)
- ✅ vLLM bound to `0.0.0.0` (not `127.0.0.1`)
- ✅ Pod is running (check RunPod console)
- ✅ Using correct URL: `https://[POD_ID]-8000.proxy.runpod.net`

---

## Testing Script

Save as `test_runpod.py`:

```python
#!/usr/bin/env python3
import requests
import sys

# Replace with your Pod ID
POD_ID = "YOUR_POD_ID_HERE"  # e.g., "abc123xyz456"
PORT = 8000
BASE_URL = f"https://{POD_ID}-{PORT}.proxy.runpod.net"

# Optional: Set if you're using authentication
API_KEY = None  # Or "your-api-key-here"

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

print(f"Testing vLLM on RunPod: {BASE_URL}")
print("=" * 60)

# Test 1: Models endpoint
print("\n1. Testing /v1/models...")
try:
    response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10)
    if response.status_code == 200:
        print("✅ Models endpoint working!")
        models = response.json()
        print(f"   Available models: {[m['id'] for m in models.get('data', [])]}")
    else:
        print(f"❌ Status: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Chat completion
print("\n2. Testing /v1/chat/completions...")
try:
    payload = {
        "model": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 20
    }
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print("✅ Chat completion working!")
        print(f"   Response: {content}")
    else:
        print(f"❌ Status: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
```

Run it:
```bash
python3 test_runpod.py
```

---

## Quick Fix Commands

**If you're getting 401 errors:**

1. **Check your vLLM startup command** in RunPod:
   ```bash
   # In RunPod, go to your pod's logs and look for the command
   # If you see --api-key, that's why you're getting 401
   ```

2. **Restart without API key:**
   - Edit your pod or template
   - Remove `--api-key` from the command
   - Restart the pod

3. **Or use the API key:**
   ```bash
   curl https://YOUR_POD_ID-8000.proxy.runpod.net/v1/models \
     -H "Authorization: Bearer sk-your-api-key-here"
   ```

---

## Best Practices for RunPod

1. **Always bind to `0.0.0.0`**
   - Not `127.0.0.1` or `localhost`
   - Docker needs to accept external connections

2. **Expose the right port**
   - Match your vLLM `--port` with RunPod's exposed port

3. **Use environment variables**
   ```dockerfile
   ENV VLLM_API_KEY=${VLLM_API_KEY:-""}
   ENV VLLM_PORT=${VLLM_PORT:-8000}
   ```

4. **Check GPU memory**
   - Use `nvidia-smi` in pod terminal
   - Adjust `--gpu-memory-utilization` if needed

5. **Monitor costs!**
   - RunPod charges by the hour
   - Stop pods when not in use

---

## Expected Performance on RunPod

| GPU | VRAM | Model Size | Speed |
|-----|------|------------|-------|
| RTX 4090 | 24GB | 3-7B | 100-150 tok/s |
| A40 | 48GB | 7-13B | 80-120 tok/s |
| A100 (40GB) | 40GB | 13B-30B | 100-200 tok/s |
| A100 (80GB) | 80GB | 30B-70B | 80-150 tok/s |

---

## Summary

**Your 401 error means:**
- ✅ vLLM is running correctly
- ✅ Port is exposed correctly
- ❌ You need to pass an API key OR remove the API key requirement

**Quick fix:**
1. Find your Pod ID in RunPod console
2. Try: `curl https://[POD_ID]-8000.proxy.runpod.net/v1/models`
3. If 401, either:
   - Add `-H "Authorization: Bearer YOUR_KEY"` to curl
   - Or restart vLLM without `--api-key` flag

You're almost there! 🚀

