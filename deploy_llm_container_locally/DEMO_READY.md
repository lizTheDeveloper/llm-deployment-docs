# LLM Docker Demo - Ready to Present

## Quick Demo Setup

### Option 1: Fast Demo (MLX - Already Running) ⚡
**Best for: Live demos, quick testing**

```bash
# Already running on port 8000!
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain Docker in one sentence"}],
    "max_tokens": 50
  }'
```

**Performance:** 20-60 tokens/second  
**Status:** ✅ Running and tested

---

### Option 2: Docker Demo (vLLM CPU - Building) 🐳
**Best for: Docker demonstrations, container workflows**

**Current Status:** Building (15-25 minutes)

**Once built, run:**
```bash
# Stop MLX server first (using same port)
pkill -f mlx_api_server

# Run vLLM in Docker
docker run -d \
  -p 8000:8000 \
  --name vllm-demo \
  vllm-cpu:demo

# Wait 30-60 seconds for model to load
sleep 60

# Test it
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is Docker?"}],
    "max_tokens": 50
  }'
```

**Performance:** 1-5 tokens/second (CPU-only in Docker)  
**Model:** TinyLlama 1.1B (faster on CPU than larger models)

---

## Demo Script

### Show Docker Container Running

```bash
# Show container status
docker ps

# Show logs
docker logs vllm-demo --tail 20

# Show resource usage
docker stats vllm-demo --no-stream
```

### Test OpenAI-Compatible API

```bash
# List models
curl http://localhost:8000/v1/models | jq

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }' | jq '.choices[0].message.content'

# Use with OpenAI Python SDK
python3 << EOF
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="TinyLlama",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
EOF
```

---

## Key Demo Points

1. **OpenAI-Compatible API** - Same API as ChatGPT/GPT-4
2. **Containerized** - Runs in Docker, portable anywhere
3. **Open Source** - No API keys, run locally
4. **Scalable** - Same container works on cloud GPUs

---

## Switching Between Demos

### Use MLX (Fast, Native)
```bash
pkill -f mlx_api_server  # Stop if running
cd deploy_llm_container_locally
./run_mlx_native.sh
```

### Use Docker (Portable, Containerized)
```bash
pkill -f mlx_api_server  # Stop MLX first
docker start vllm-demo   # Or use docker run command above
```

---

## Troubleshooting

**Port 8000 already in use:**
```bash
# Stop MLX server
pkill -f mlx_api_server

# Or stop Docker container
docker stop vllm-demo
```

**Docker build failed:**
```bash
# Check build logs
tail -100 /tmp/vllm_build.log

# Retry with more memory
docker build --memory=8g --progress=plain -t vllm-cpu:demo -f Dockerfile.vllm-cpu-fixed .
```

**Container exits immediately:**
```bash
# Check logs
docker logs vllm-demo

# Run in foreground to see errors
docker run --rm -p 8000:8000 vllm-cpu:demo
```

---

## Performance Comparison

| Deployment | Tokens/Sec | Docker? | Demo Use Case |
|------------|------------|---------|---------------|
| **MLX Native** | 20-60 | ❌ | Fast demo, development |
| **vLLM CPU Docker** | 1-5 | ✅ | Docker demo, portability |
| **vLLM GPU (AWS)** | 100-200+ | ✅ | Production showcase |

---

## Production Deployment Demo

To show cloud deployment:

```bash
# SSH to AWS EC2 with GPU
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Run same container with GPU
docker run -d --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct

# Test from local machine
curl http://YOUR_EC2_IP:8000/v1/models
```

See: `docs/CLOUD_GPU_DEPLOYMENT_GUIDE.md` for full setup

