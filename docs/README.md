# LLM Deployment Documentation

Production-ready guides for deploying large language models at any scale, from single GPU instances to enterprise Kubernetes clusters.

## Start Here

### New to LLM Deployment?
→ **[Lab 8: Production vLLM Deployment](LAB_8_VLLM_DEPLOYMENT.md)** - Deploy your first LLM with Docker in 30 minutes

### Want Tool Calling & Multi-Service Architecture?
→ **[Lab 9: FastAPI Tool Calling](LAB_9_TOOL_CALLING.md)** - Build a two-tier system (FastAPI + vLLM) in 45 minutes

### Deploying to Cloud?
→ **[Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md)** - AWS/GCP/Azure deployment for Mac users

### Scaling to Production?
→ **[Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)** - Kubernetes, multi-model serving, 70B+ models

### Learning from Others?
→ **[Real-World Case Studies](REAL_WORLD_DEPLOYMENT_BLOGS.md)** - 50+ blog posts from Salesforce, Meta, Anthropic

---

## Reference Files

Production code examples from the course labs:
- `Dockerfile.vllm` - Production Docker image with vLLM v1 engine
- `tool_orchestrator.py` - FastAPI orchestration layer for tool calling
- `docker-compose.yml` - Multi-service deployment example

---

## Common Deployment Scenarios

| Your Goal | Start With | What You'll Build |
|-----------|------------|-------------------|
| First LLM deployment | [Lab 8: vLLM Deployment](LAB_8_VLLM_DEPLOYMENT.md) | Docker container with OpenAI API (30 min) |
| Tool calling system | [Lab 9: Tool Calling](LAB_9_TOOL_CALLING.md) | FastAPI + vLLM two-tier architecture (45 min) |
| Cloud deployment | [Cloud GPU Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md) | Single 7B model on AWS/GCP/Azure |
| Multiple models | [Enterprise-Scale](ENTERPRISE_SCALE_DEPLOYMENT.md) | Multi-model Kubernetes deployment |
| Large model (70B) | [Enterprise-Scale](ENTERPRISE_SCALE_DEPLOYMENT.md) | Tensor parallelism on 4× A100 GPUs |
| Massive model (405B) | [Enterprise-Scale](ENTERPRISE_SCALE_DEPLOYMENT.md) | Pipeline + tensor parallelism (multi-node) |
| Customer-specific models | [Enterprise-Scale](ENTERPRISE_SCALE_DEPLOYMENT.md) | LoRA multi-tenancy (2,000+ adapters) |
| Research case studies | [Real-World Blogs](REAL_WORLD_DEPLOYMENT_BLOGS.md) | Learn from Salesforce, Meta, Anthropic |

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
