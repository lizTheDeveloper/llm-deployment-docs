# Local LLM Deployment Options for Mac

This directory contains different approaches to run LLMs locally on Mac.

## ⚠️ Important: Docker + Apple Silicon Limitations

**Docker on Mac cannot access Metal GPU acceleration.** All Docker containers run in a Linux VM and can only use CPU. This means:

- ❌ `Dockerfile.vllm` - Original CUDA version (requires NVIDIA GPU, doesn't work on Mac)
- ⚠️ `Dockerfile.vllm-cpu` - vLLM CPU mode (works but extremely slow: 1-5 tokens/sec)
- ❌ `Dockerfile.mlx` - Won't work in Docker (MLX requires native macOS, not Linux VM)

## ✅ Recommended: Run MLX-LM Natively (No Docker)

**Best performance on M1/M2/M3 Macs** - Uses Metal GPU acceleration:

```bash
# Create virtual environment
python3 -m venv mlx-env
source mlx-env/bin/activate

# Install MLX-LM
pip install mlx-lm fastapi uvicorn[standard]

# Run OpenAI-compatible server
python mlx_api_server.py \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 0.0.0.0 \
  --port 8000
```

**Performance:** 20-60 tokens/second on M3 (same API as vLLM!)

**Test it:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Alternative: llama.cpp in Docker (CPU-only)

If you need Docker for testing deployment workflows:

```bash
# Download GGUF model
mkdir -p ~/llama-models
wget https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -O ~/llama-models/qwen-3b.gguf

# Run llama.cpp server (OpenAI-compatible)
docker run -d \
  --name llama-server \
  -p 8000:8080 \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen-3b.gguf \
  --host 0.0.0.0 \
  --port 8080

# Test
curl http://localhost:8000/v1/models
```

**Performance:** 5-15 tokens/second on M3 (CPU only in Docker)

## ⚠️ If You Really Need vLLM CPU Mode

Only use this for testing deployment scripts, not actual inference:

```bash
# Build CPU-only vLLM (takes 20-30 minutes)
cd deploy_llm_container_locally
./build_vllm_cpu.sh

# After build completes, run the container
docker run -d \
  --name vllm-cpu-server \
  -p 8000:8000 \
  vllm-cpu:latest

# Test it
python test_api.py

# This will work but expect 1-5 tokens/second
```

**Build requirements:**
- 20-30 minutes build time
- 4-8GB disk space
- cmake, ninja-build, C++ compiler (included in Dockerfile)

## Performance Comparison on M3 Mac

| Method | Docker? | GPU? | Speed | Use Case |
|--------|---------|------|-------|----------|
| **MLX-LM (native)** | ❌ No | ✅ Metal | 20-60 tok/s | **Recommended for Mac** |
| llama.cpp (native) | ❌ No | ✅ Metal | 15-40 tok/s | Good alternative |
| llama.cpp (Docker) | ✅ Yes | ❌ CPU only | 5-15 tok/s | Testing deployments |
| vLLM CPU (Docker) | ✅ Yes | ❌ CPU only | 1-5 tok/s | Not recommended |
| vLLM CUDA | ✅ Yes | ❌ N/A | Won't run | **Deploy to AWS instead** |

## Production Deployment

For actual GPU-accelerated vLLM, deploy to cloud:

```bash
# See docs/CLOUD_GPU_DEPLOYMENT_GUIDE.md
# Quick AWS deployment:
# 1. Launch g5.xlarge EC2 instance
# 2. SSH in and run:
docker run -d --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model unsloth/Qwen2.5-7B-Instruct
```

**Cost:** $1/hour on AWS g5.xlarge
**Speed:** 100-200 tokens/second with GPU

## Available Models for MLX

Browse MLX-optimized models: https://huggingface.co/mlx-community

Popular choices:
- `mlx-community/Qwen2.5-3B-Instruct-4bit` (2.3GB)
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (4.8GB)
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (2.3GB)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (4.5GB)

