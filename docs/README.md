# LLM Deployment Documentation

Production-ready guides for deploying large language models at any scale, from single GPU instances to enterprise Kubernetes clusters.

## Which Guide Should I Read?

### Just Getting Started?
→ **[Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md)** - Deploy your first LLM to AWS, GCP, or Azure in 15 minutes

### Deploying at Scale?
→ **[Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)** - Multi-model serving, 70B+ models, LoRA multi-tenancy

### Looking for Real-World Examples?
→ **[Real-World Case Studies](REAL_WORLD_DEPLOYMENT_BLOGS.md)** - 50+ blog posts from companies like Salesforce, Meta, and Anthropic

---

## Lab Files

This repository includes production-ready code from Labs 8 and 9:

**Lab 8 - vLLM Deployment**
- `Dockerfile.vllm` - Production Docker image with vLLM v1 engine

**Lab 9 - Tool Calling with FastAPI**
- `tool_orchestrator.py` - FastAPI orchestration layer for tool calling
- `docker-compose.yml` - Multi-service deployment (vLLM + orchestrator)
- `Dockerfile.orchestrator` - Docker image for FastAPI layer

### Quick Start

```bash
# Single vLLM service (Lab 8)
docker build -f Dockerfile.vllm -t my-llm-service .
docker run --gpus all -p 8000:8000 my-llm-service

# Multi-service with tool calling (Lab 9)
docker-compose up
```

---

## Deployment Guides Overview

### Cloud GPU Deployment Guide

**For:** Mac users, first-time cloud deployers, single-instance staging environments

**Covers:**
- Step-by-step AWS/GCP/Azure GPU instance setup
- Deploying vLLM and llama.cpp containers
- SSH, Docker, and API configuration
- Cost optimization strategies
- Mac local testing with Ollama

### Enterprise-Scale Deployment

**For:** Production deployments, multi-model serving, enterprise workloads

**Covers:**
- Multi-model serving with Kubernetes
- 70B models with tensor parallelism (4× A100)
- 405B models with pipeline parallelism (multi-node)
- LoRA multi-tenancy for customer-specific models
- Autoscaling, monitoring, and cost optimization

### Real-World Deployment Blogs

**For:** Learning from production deployments, researching vendor solutions

**Includes:**
- vLLM Production Stack and v1 engine case studies
- Cloud platform guides (AWS Bedrock, Vertex AI, Azure)
- Enterprise deployments (Salesforce, FactSet, Ford Direct)
- Multi-tenant LoRA serving (Punica, SageMaker)
- Infrastructure at scale (Meta Llama 3.1 on 16K H100s)

---

## Quick Reference Table

| Your Goal | Guide | Key Concepts |
|-----------|-------|--------------|
| Deploy first model to cloud | Cloud GPU Guide | AWS EC2, SSH, vLLM container |
| Run 7B model in production | Cloud GPU Guide | A10G instance, monitoring |
| Serve multiple 7B models | Enterprise-Scale | Multi-model, Kubernetes |
| Deploy 70B model | Enterprise-Scale | Tensor parallelism, A100 |
| Deploy 405B model | Enterprise-Scale | Pipeline parallelism, H100, FP8 |
| Customer-specific LoRAs | Enterprise-Scale | S-LoRA, multi-tenancy |
| Tool calling / RAG | Lab 9 files | FastAPI, two-tier architecture |
| Learn from case studies | Real-World Blogs | Vendor comparisons, benchmarks |

---

## Additional Resources

**Official Documentation**
- [vLLM Docs](https://docs.vllm.ai/) - Official vLLM documentation
- [vLLM v1 Release](https://blog.vllm.ai/2025/01/13/v1.html) - Latest performance improvements

**Cloud Providers**
- [AWS GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [GCP GPU VMs](https://cloud.google.com/compute/docs/gpus)
- [Azure GPU VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)

### Support and Community
- **GitHub Issues**: Report bugs or request features
- **vLLM Slack**: Join the community for production deployment discussions
- **Course Updates**: Check `REAL_WORLD_DEPLOYMENT_BLOGS.md` for latest industry blog posts
