# Deployment Files

This directory contains production deployment configuration files, enterprise-scale guides, and real-world case study references used in Labs 8 and 9.

## Files

### Lab 8 - vLLM Deployment

- **`Dockerfile.vllm`** - Production Docker image for deploying models with vLLM
  - Uses official vLLM OpenAI-compatible image
  - Configured for GPU deployment
  - Includes health checks for container orchestration
  - Auto-optimized with vLLM v1 engine

### Lab 9 - vLLM + FastAPI Tool Calling

- **`tool_orchestrator.py`** - FastAPI orchestration layer for tool calling
  - Handles tool calling logic (weather API, calculator, etc.)
  - Proxies requests to vLLM for inference
  - OpenAI-compatible API endpoints
  - Runs on port 8001

- **`docker-compose.yml`** - Multi-service deployment configuration
  - vLLM service: GPU-accelerated inference (port 8000)
  - Orchestrator service: FastAPI tool calling layer (port 8001)
  - Configured for production deployment with Docker Compose

- **`Dockerfile.orchestrator`** - Docker image for FastAPI orchestrator
  - Lightweight Python 3.11 image
  - Includes FastAPI, uvicorn, httpx dependencies
  - Runs tool_orchestrator.py

## Usage

### Lab 8: Single vLLM Service

```bash
# Build and run vLLM service
docker build -f Dockerfile.vllm -t my-llm-service:latest .
docker run --gpus all -p 8000:8000 my-llm-service:latest
```

### Lab 9: Multi-Service Architecture

```bash
# Start both vLLM and orchestrator services
docker-compose up

# Services will be available at:
# - vLLM: http://localhost:8000
# - Orchestrator: http://localhost:8001
```

## Production Deployment

### AWS ECS

Deploy as separate services:
1. **vLLM Service**: GPU-enabled tasks (g5.xlarge or larger)
2. **Orchestrator Service**: CPU tasks (can scale horizontally)

### Kubernetes

Use separate deployments:
- vLLM: GPU NodePool with vertical scaling
- Orchestrator: CPU NodePool with horizontal scaling (HPA)

### Monitoring

vLLM exposes Prometheus metrics at `/metrics`:
- Request queue depth (for autoscaling triggers)
- GPU memory utilization
- Token throughput
- Time to first token (TTFT)
- Time per output token (TPOT)

## Architecture Benefits

**Two-Tier Design** (FastAPI + vLLM):
- **Separation of concerns**: Business logic vs inference
- **Independent scaling**: Scale each tier based on load
- **Cost optimization**: GPU only for inference, CPU for orchestration
- **Flexibility**: Update tool logic without redeploying vLLM

This is how production systems at Anthropic, OpenAI, and Google structure their tool-calling capabilities.

## Enterprise-Scale Deployment Guides

### ENTERPRISE_SCALE_DEPLOYMENT.md

**Comprehensive guide for deploying LLMs at Salesforce scale**

Topics covered:
- **Multi-Model Serving**: Deploy multiple 7B models with vLLM Production Stack and Kubernetes
- **70B Model Deployment**: Tensor parallelism across 4× A100 GPUs, quantization strategies
- **405B Model Deployment**: Pipeline parallelism, multi-node setup, FP8 quantization on H100
- **LoRA Multi-Tenancy**: S-LoRA for serving 2,000+ customer-specific adapters
- **Production Architecture Patterns**: Two-tier, direct vLLM, multi-tier with LoRA
- **Cost Optimization**: GPU right-sizing, spot instances, autoscaling, prefix caching

**Use this guide when:**
- Deploying multiple models simultaneously
- Scaling to enterprise workloads (Salesforce-level)
- Implementing customer-specific fine-tuning with LoRA adapters
- Optimizing infrastructure costs at scale

### REAL_WORLD_DEPLOYMENT_BLOGS.md

**Curated collection of 50+ authoritative blog posts and case studies**

Categories:
- **vLLM Production Deployments**: Official vLLM blog posts on v1 engine, Production Stack, AIBrix
- **Cloud Platform Deployments**: AWS Bedrock, Google Vertex AI, Microsoft Azure, Oracle Cloud
- **Enterprise Case Studies**: Salesforce (30% faster, 40% cost savings), FactSet (55%→85% accuracy), Ford Direct, Corning, Burberry, Dynamo AI
- **Multi-Tenant LoRA Serving**: Punica (12x throughput), AWS SageMaker, CaraServe, LoRA-Inlaid
- **Infrastructure at Scale**: Meta Llama 3.1 (16K H100 GPUs), Anthropic Claude multi-platform deployment
- **AI Platform Vendors**: Together AI, Anyscale, Databricks Mosaic AI

**Use this document when:**
- Learning from real-world production deployments
- Researching vendor solutions and performance benchmarks
- Justifying architecture decisions with case studies
- Staying current on latest deployment techniques (updated monthly)

### CLOUD_GPU_DEPLOYMENT_GUIDE.md

**Practical cloud deployment guide for Mac users and developers**

Topics covered:
- AWS EC2 GPU instance setup (g5.xlarge, p4d, p5)
- GCP Compute Engine GPU setup (L4, A100)
- Azure GPU VM setup (V100, A100)
- Deploying vLLM containers to cloud instances
- Deploying llama.cpp containers to cloud instances
- Cost comparison and optimization strategies
- Mac local testing with CPU-only inference (Ollama, llama.cpp)

**Use this guide when:**
- Working from a Mac (no local GPU available)
- Deploying to a single cloud GPU instance for staging/testing
- Learning cloud GPU deployment before scaling to Kubernetes
- Need step-by-step SSH, Docker, and API setup instructions
- Want to understand costs and choose the right instance type

## Additional Resources

### Documentation Links
- **vLLM Official Docs**: https://docs.vllm.ai/
- **vLLM v1 Blog**: https://blog.vllm.ai/2025/01/13/v1.html
- **AWS GPU Instances**: https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing
- **GCP GPU VMs**: https://cloud.google.com/compute/docs/gpus
- **Azure GPU VMs**: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu

### Quick Reference

**Choose your deployment path:**

| Scenario | Recommended Guide | Key Files |
|----------|-------------------|-----------|
| Mac user, first cloud deployment | CLOUD_GPU_DEPLOYMENT_GUIDE.md | - |
| Single 7B model on AWS/GCP/Azure | CLOUD_GPU_DEPLOYMENT_GUIDE.md | Dockerfile.vllm |
| Single 7B model (production) | Lab 8 files | Dockerfile.vllm |
| Multiple 7B models | ENTERPRISE_SCALE_DEPLOYMENT.md (Multi-Model) | docker-compose.yml |
| Tool calling / RAG | Lab 9 files | tool_orchestrator.py, docker-compose.yml |
| 70B model | ENTERPRISE_SCALE_DEPLOYMENT.md (70B) | Kubernetes YAML examples |
| 405B model | ENTERPRISE_SCALE_DEPLOYMENT.md (405B) | Multi-node examples |
| Multi-tenant (customer LoRAs) | ENTERPRISE_SCALE_DEPLOYMENT.md (LoRA) | - |
| Research case studies | REAL_WORLD_DEPLOYMENT_BLOGS.md | - |

### Performance Expectations (vLLM v1 Engine)

| Model Size | GPU | Throughput | TTFT | Cost/Hour (AWS) |
|------------|-----|------------|------|-----------------|
| 7B (FP16) | A10G (24GB) | 50-100 tok/s | 50-100ms | $1.01 |
| 7B (INT4) | T4 (16GB) | 50-70 tok/s | 100-150ms | $0.35 |
| 70B (FP16) | 4× A100 40GB | 30-50 tok/s | 200-400ms | $13-16 |
| 70B (AWQ) | 2× A100 80GB | 40-60 tok/s | 150-300ms | $10-13 |
| 405B (FP8) | 8× H100 80GB | 10-30 tok/s | 1-3s | $32-40 |

### Support and Community
- **GitHub Issues**: Report bugs or request features
- **vLLM Slack**: Join the community for production deployment discussions
- **Course Updates**: Check `REAL_WORLD_DEPLOYMENT_BLOGS.md` for latest industry blog posts

---

## Deploy These Docs as a Website

**This directory is GitBook-ready!** Deploy as a professional documentation site in 5 minutes.

See **[GITBOOK_DEPLOYMENT.md](GITBOOK_DEPLOYMENT.md)** for:
- GitBook.com setup (easiest, free)
- MkDocs Material (gorgeous design)
- GitHub Pages deployment
- Custom domain setup

**Quick start:**
```bash
# Push to GitHub
gh repo create llm-deployment-docs --public --source=. --push

# Import to GitBook.com
# Sign up → Import from GitHub → Select deployment_files/

# Done! Live at: https://your-username.gitbook.io/llm-deployment-guide/
```

All markdown files are platform-agnostic and work with GitBook, MkDocs, Docusaurus, or any documentation tool.
