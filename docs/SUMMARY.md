# Table of Contents

## Getting Started

* [Introduction](README.md)
* [📊 Course Slides & Labs](https://docs.google.com/presentation/d/1-FTmWgVct1Ydkwvyy8ZR-mFl7KGbH-TzZRZvfMk5aRo/edit?slide=id.g39bdb786812_0_79)

## Lab Notebooks

### Labs 1-2: Foundations (Mac Compatible)
* [Lab 1: Keras Quick Refresher](../lab_notebooks/Lab1_Keras_Quick_Refresher.ipynb)
* [Lab 2: GradientTape Refresher](../lab_notebooks/Lab2_GradientTape_Refresher.ipynb)

### Labs 3-7: Unsloth Optimization (Requires Colab)
* [Lab 3: PyTorch Fundamentals](../lab_notebooks/Lab3_PyTorch_Fundamentals.ipynb)
* [Lab 4: Hello Unsloth](../lab_notebooks/Lab4_Hello_Unsloth.ipynb)
* [Lab 5: Knowledge Distillation with SQuAD](../lab_notebooks/Lab5_Distillation_Unsloth_SQuAD.ipynb)
* [Lab 6: Model Pruning with SST-2](../lab_notebooks/Lab6_Pruning_Unsloth_SST2.ipynb)
* [Lab 7: Quantization with IMDB](../lab_notebooks/Lab7_Quantization_Unsloth_IMDB.ipynb)

### Labs 8-9: Deployment (Mac Compatible)
* [Lab 8: FastAPI OpenAI-Compatible API](../lab_notebooks/Lab8_Deployment_OpenAI_Compatible_FastAPI.ipynb)
* [Lab 9: FastAPI Tool Calling with vLLM](../lab_notebooks/Lab9_Deployment_OpenAI_Compatible_FastAPI_With_Tool_Calling.ipynb)

## Research Papers

* [📄 DeepSeek-R1: Knowledge Distillation](DEEPSEEK_R1_DISTILLATION.md)
  * Theoretical Foundation for Lab 5
  * Distillation Methodology (Section 2.4)
  * Lab 5 & Lab 6 Context

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

## Solutions

* [✅ Complete Lab Solutions](SOLUTIONS.md)
  * Solution Notebooks (Labs 1-9)
  * Python Test Versions (Mac M3 Compatible)

## Reference

* [llama.cpp Docker (Linux Only)](LLAMA_CPP_DOCKER_GUIDE_LINUX_ONLY.md)
