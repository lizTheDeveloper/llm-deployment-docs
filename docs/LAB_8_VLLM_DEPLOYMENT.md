# Lab 8: Production vLLM Deployment

Deploy a production-ready LLM inference server using vLLM's OpenAI-compatible API with Docker on AWS.

**Time:** 30 minutes (after AWS setup)
**Prerequisites:** AWS EC2 GPU instance with Docker (see setup below)
**What You'll Build:** Production Docker container serving a 7B model with health checks and monitoring

**Don't have an AWS GPU instance yet?** → Start with [Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md) to set up your AWS EC2 g5.xlarge instance (15 minutes)

---

## What You'll Learn

- Package an LLM for production deployment with Docker
- Configure vLLM v1 engine for optimal performance
- Set up health checks for container orchestration (ECS/Kubernetes)
- Deploy to AWS EC2 GPU instance
- Test the OpenAI-compatible API from your local machine

---

## Architecture

```
Client (OpenAI SDK/curl)
    ↓
vLLM Container (Port 8000)
    ↓
GPU Inference (vLLM v1 engine)
```

**Key Features:**
- OpenAI-compatible API endpoint
- Automatic request batching
- GPU memory optimization (90% utilization)
- Health check endpoint for Kubernetes/ECS
- vLLM v1 engine with auto-optimization

---

## Step 1: Prepare Your Model

### Option A: Use a Hugging Face Model (Recommended for Learning)

You don't need to copy the model into the container. vLLM will download it automatically:

```bash
# No preparation needed! vLLM downloads from Hugging Face
```

### Option B: Use Your Own Fine-Tuned Model

If you have a custom model, create a directory structure:

```bash
mkdir -p vllm_model
# Copy your model files to vllm_model/
cp -r /path/to/your/model/* vllm_model/
```

---

## Step 2: Create the Dockerfile

Create `Dockerfile.vllm`:

```dockerfile
# Use official vLLM image (includes CUDA, PyTorch, and all dependencies)
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy your fine-tuned model (skip if using Hugging Face models)
# COPY ./vllm_model /app/model

# Expose vLLM port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start vLLM OpenAI-compatible server (v1 engine)
# Note: Don't add --enable-chunked-prefill or --num-scheduler-steps
# These force v0 fallback. v1 engine (default) auto-optimizes these.
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-7B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--dtype", "float16", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.9"]
```

**Configuration Explained:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | Model name (Hugging Face) or local path |
| `--dtype` | `float16` | Half precision (16GB VRAM for 7B model) |
| `--max-model-len` | `4096` | Maximum context length |
| `--gpu-memory-utilization` | `0.9` | Use 90% of GPU memory for KV cache |
| `--host` | `0.0.0.0` | Listen on all interfaces |
| `--port` | `8000` | Standard vLLM port |

---

## Step 3: Build the Docker Image

```bash
# Build the image
docker build -f Dockerfile.vllm -t my-llm-service:latest .

# This will take 2-5 minutes
# The vLLM image is ~10GB, model download ~14GB for 7B model
```

---

## Step 4: Deploy to AWS EC2

### Prerequisites: AWS EC2 Instance Setup

If you haven't set up your AWS EC2 instance yet, follow this quick start:

1. **Launch EC2 instance**: AWS Console → EC2 → Launch `g5.xlarge` (Deep Learning AMI)
2. **Download key**: Save `llm-server-key.pem`
3. **SSH into instance**: `ssh -i llm-server-key.pem ubuntu@YOUR_INSTANCE_IP`

**Full AWS setup guide:** [Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md)

### Run the Container on AWS

SSH into your EC2 instance, then run:

```bash
# On your AWS EC2 instance
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# Check logs
docker logs -f vllm-server

# Wait for this message:
# "Application startup complete"
# "Uvicorn running on http://0.0.0.0:8000"
```

**Note:** Using the vLLM image directly is faster than building your own Dockerfile for initial testing.

### Alternative: Local GPU or Other Clouds

If you have a local NVIDIA GPU or prefer GCP/Azure:

```bash
# Same docker run command works on any GPU instance with Docker + nvidia-docker
docker run -d --name vllm-server --gpus all -p 8000:8000 ...
```

---

## Step 5: Test the API from Your Local Machine

Now test the deployed API from your local Mac/PC. Replace `YOUR_INSTANCE_IP` with your EC2 public IP.

### Health Check

```bash
# From your local terminal (Mac/PC)
export AWS_IP=54.123.45.67  # Replace with your EC2 public IP

curl http://${AWS_IP}:8000/health

# Response:
# {"status": "ok"}
```

### List Available Models

```bash
curl http://${AWS_IP}:8000/v1/models

# Response:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "Qwen/Qwen2.5-7B-Instruct",
#       "object": "model",
#       "owned_by": "vllm",
#       ...
#     }
#   ]
# }
```

### Generate a Completion

```bash
curl http://${AWS_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Using the OpenAI Python SDK (from Your Mac)

```python
from openai import OpenAI

# Point to your AWS vLLM server
client = OpenAI(
    base_url="http://54.123.45.67:8000/v1",  # Replace with your EC2 IP
    api_key="dummy"  # vLLM doesn't require authentication
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about cloud computing"}
    ]
)

print(response.choices[0].message.content)
```

**Troubleshooting connection refused?** Check your EC2 Security Group allows inbound traffic on port 8000 from your IP.

---

## Step 6: Monitor Performance on AWS

### GPU Utilization on EC2

```bash
# SSH into your AWS EC2 instance
ssh -i llm-server-key.pem ubuntu@YOUR_INSTANCE_IP

# Check GPU utilization
nvidia-smi

# You should see:
# - GPU memory usage (14-16GB for 7B FP16 model)
# - GPU utilization (50-100% during inference)
# - Temperature and power usage
```

### vLLM Metrics (from your local machine)

```bash
# From your Mac/PC - vLLM exposes Prometheus metrics
curl http://${AWS_IP}:8000/metrics

# Key metrics for production monitoring:
# - vllm:num_requests_running       # Active requests
# - vllm:time_to_first_token_seconds # Latency
# - vllm:time_per_output_token_seconds # Throughput
# - vllm:gpu_cache_usage_perc        # Memory usage
```

**AWS CloudWatch Integration:** For production, export these metrics to CloudWatch using the CloudWatch Agent on your EC2 instance.

---

## AWS Production Deployment Checklist

Before deploying to production on AWS:

- [ ] **Security Groups**: Restrict port 8000 to specific IPs (not 0.0.0.0/0)
- [ ] **Load Balancer**: Set up AWS Application Load Balancer with HTTPS (ACM certificate)
- [ ] **AWS ECS**: Deploy container to ECS Fargate or EC2 with GPU task definitions
- [ ] **CloudWatch**: Configure CloudWatch Logs and custom metrics
- [ ] **Auto Scaling**: Set up ECS auto scaling based on request queue depth
- [ ] **API Gateway**: Add AWS API Gateway for rate limiting and API keys
- [ ] **Authentication**: Implement AWS Cognito or API Gateway auth
- [ ] **Load Testing**: Use AWS Load Testing tools or Locust to validate performance

**Next Level**: See [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md) for Kubernetes on AWS EKS

---

## Troubleshooting

### "CUDA out of memory" Error

**Problem:** GPU doesn't have enough VRAM

**Solutions:**
```bash
# Option 1: Reduce max model length
--max-model-len 2048

# Option 2: Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Option 3: Use quantization (INT8 or AWQ)
--quantization awq
```

### "nvidia-smi: command not found"

**Problem:** NVIDIA drivers not installed

**Solution:** See [Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md) for driver installation

### Model Download is Slow

**Problem:** Hugging Face download is slow from your region

**Solutions:**
```bash
# Option 1: Pre-download model on host
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./vllm_model

# Option 2: Use Hugging Face mirror (China)
-e HF_ENDPOINT=https://hf-mirror.com
```

---

## Next Steps

**Completed Lab 8?** Move on to:

→ **[Lab 9: FastAPI Tool Calling](LAB_9_TOOL_CALLING.md)** - Add tool calling with a FastAPI orchestration layer

→ **[Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)** - Deploy to Kubernetes with autoscaling

→ **[Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md)** - Deploy to AWS/GCP/Azure

---

## Key Takeaways

✅ vLLM provides an OpenAI-compatible API out of the box
✅ vLLM v1 engine auto-optimizes performance (no manual tuning)
✅ Docker makes deployment consistent across environments
✅ Health checks enable container orchestration (Kubernetes/ECS)
✅ 90% GPU memory utilization maximizes throughput

**You now have a production-ready LLM inference server!**

---

## References

- **vLLM v1 Blog:** [https://blog.vllm.ai/2025/01/27/v1-alpha-release.html](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) - vLLM V1 architecture and performance improvements
- **vLLM Documentation:** [https://docs.vllm.ai/](https://docs.vllm.ai/) - Official vLLM documentation
- **vLLM GitHub Releases:** [https://github.com/vllm-project/vllm/releases](https://github.com/vllm-project/vllm/releases) - Latest releases (v0.11.1 as of October 2025)
- **OpenAI API Specification:** [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference) - API compatibility reference

**Last Updated:** October 2025 | **vLLM Version:** v0.11.1 (October 2025)
