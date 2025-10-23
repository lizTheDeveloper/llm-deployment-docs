# LLM Deployment Documentation

Production-ready guides for deploying large language models at any scale, from single GPU instances to enterprise Kubernetes clusters.

## Start Here

### Course Materials
→ **[📊 Course Slides & All Labs](https://docs.google.com/presentation/d/1-FTmWgVct1Ydkwvyy8ZR-mFl7KGbH-TzZRZvfMk5aRo/edit?slide=id.g39bdb786812_0_79)** - All lab instructions, exercises, and walkthroughs

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

## Lab Notebooks

Access all course lab notebooks:

### Labs 1-2: Foundations (Mac Compatible)
- [Lab 1: Keras Quick Refresher](../lab_notebooks/Lab1_Keras_Quick_Refresher.ipynb)
- [Lab 2: GradientTape Refresher](../lab_notebooks/Lab2_GradientTape_Refresher.ipynb)

### Labs 3-7: Unsloth Optimization (Requires Colab)
- [Lab 3: PyTorch Fundamentals](../lab_notebooks/Lab3_PyTorch_Fundamentals.ipynb)
- [Lab 4: Hello Unsloth](../lab_notebooks/Lab4_Hello_Unsloth.ipynb)
- [Lab 5: Knowledge Distillation with SQuAD](../lab_notebooks/Lab5_Distillation_Unsloth_SQuAD.ipynb)
- [Lab 6: Model Pruning with SST-2](../lab_notebooks/Lab6_Pruning_Unsloth_SST2.ipynb)
- [Lab 7: Quantization with IMDB](../lab_notebooks/Lab7_Quantization_Unsloth_IMDB.ipynb)

### Labs 8-9: Deployment (Mac Compatible)
- [Lab 8: FastAPI OpenAI-Compatible API](../lab_notebooks/Lab8_Deployment_OpenAI_Compatible_FastAPI.ipynb) | [Walkthrough Guide](LAB_8_VLLM_DEPLOYMENT.md)
- [Lab 9: FastAPI Tool Calling with vLLM](../lab_notebooks/Lab9_Deployment_OpenAI_Compatible_FastAPI_With_Tool_Calling.ipynb) | [Walkthrough Guide](LAB_9_TOOL_CALLING.md)

**[✅ View Complete Solutions →](SOLUTIONS.md)**

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
