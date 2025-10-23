# Table of Contents

## Getting Started

* [Introduction](README.md)

## Deployment Guides

* [Cloud GPU Deployment Guide](CLOUD_GPU_DEPLOYMENT_GUIDE.md)
  * AWS EC2 GPU Setup
  * GCP Compute Engine GPU Setup
  * Azure GPU VM Setup
  * Deploying vLLM Containers
  * Deploying llama.cpp Containers
  * Cost Comparison & Optimization
  * Mac Local Testing (CPU-Only)

* [GKE kubectl Commands Guide](GKE_KUBECTL_COMMANDS.md)
  * GKE Autopilot Cluster Setup
  * kubectl Deployment Commands
  * Monitoring & Troubleshooting
  * Scaling & Cost Optimization

* [GCP Test Deployment](GCP_TEST_DEPLOYMENT.md)
  * GCP vs AWS Comparison
  * Small-Scale Deployment (400 Users)
  * L4 GPU Setup
  * Alternative Deep Learning VM

* [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)
  * Multi-Model Serving (7B Models)
  * 70B Model Deployment
  * 405B Model Deployment
  * LoRA Multi-Tenancy
  * Production Architecture Patterns
  * Cost Optimization Strategies

* [Real-World Deployment Blogs](REAL_WORLD_DEPLOYMENT_BLOGS.md)
  * vLLM Production Deployments
  * Cloud Platform Deployments
  * Enterprise Case Studies
  * Multi-Tenant LoRA Serving
  * Infrastructure at Scale
  * AI Platform Vendors

## Lab Files

* [Dockerfile.vllm](Dockerfile.vllm)
* [Dockerfile.orchestrator](Dockerfile.orchestrator)
* [docker-compose.yml](docker-compose.yml)
* [tool_orchestrator.py](tool_orchestrator.py)
* [gcp-quickstart.sh](gcp-quickstart.sh)

## Reference

* [llama.cpp Docker (Linux Only)](LLAMA_CPP_DOCKER_GUIDE_LINUX_ONLY.md)
