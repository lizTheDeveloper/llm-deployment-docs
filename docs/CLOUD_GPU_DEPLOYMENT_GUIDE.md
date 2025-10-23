# Cloud GPU Deployment Guide

Deploy your first LLM to the cloud in 15 minutes. This guide walks you through setting up a cloud GPU instance and running vLLM or llama.cpp containers.

**Target Audience:** Mac users, developers without local GPUs, teams deploying staging environments

---

## Quick Start: AWS in 5 Steps

**Fastest path from zero to running LLM API:**

1. **Launch Instance:** AWS Console → EC2 → Launch `g5.xlarge` (Deep Learning AMI)
2. **Download Key:** Save `llm-server-key.pem` from AWS
3. **SSH In:** `ssh -i llm-server-key.pem ubuntu@YOUR_INSTANCE_IP`
4. **Run vLLM:** `docker run -d --gpus all -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen2.5-7B-Instruct`
5. **Test:** `curl http://YOUR_INSTANCE_IP:8000/v1/models`

**Total time:** ~15 minutes | **Cost:** $1.01/hour | **Result:** Production-ready LLM API

*See [AWS EC2 GPU Instance Setup](#aws-ec2) for detailed instructions.*

---

## Table of Contents

1. [Quick Start: AWS in 5 Steps](#quick-start-aws-in-5-steps)
2. [AWS EC2 GPU Instance Setup](#aws-ec2)
3. [GCP Compute Engine GPU Setup](#gcp-compute)
4. [Azure GPU VM Setup](#azure-vm)
5. [Deploying vLLM Container](#deploy-vllm)
6. [Deploying llama.cpp Container](#deploy-llama-cpp)
7. [Cost Comparison & Recommendations](#cost-comparison)
8. [Mac Local Testing (CPU-Only)](#mac-local-testing)
9. [Why Cloud Deployment?](#why-cloud)

---

## Why Cloud Deployment? {#why-cloud}

### Mac Limitations

- **No NVIDIA GPUs**: Macs use Apple Silicon (M1/M2/M3) or AMD GPUs
- **Docker GPU passthrough**: Not supported for NVIDIA CUDA on Mac
- **Memory constraints**: Even Mac Studio (192GB) can't run 70B+ models efficiently
- **No production deployment**: Local Mac is for development only

### Cloud Advantages

- **GPU access**: NVIDIA A100, H100, A10G, T4 available on-demand
- **Scalability**: Start with 1 GPU, scale to 100s
- **Production-ready**: Load balancers, autoscaling, monitoring
- **Cost-effective**: Pay only for what you use ($0.35-$40/hour)

### Deployment Strategy

```
Local Mac (Dev) ──→ Cloud GPU Instance (Staging) ──→ Kubernetes Cluster (Production)
   ↓                        ↓                              ↓
Test code           Single GPU testing              Multi-GPU, autoscaling
CPU inference       Full model performance          Enterprise workloads
```

---

## AWS EC2 GPU Instance Setup {#aws-ec2}

### Step 1: Choose Your Instance Type

| Instance Type | GPU | VRAM | Use Case | Cost/Hour |
|---------------|-----|------|----------|-----------|
| g5.xlarge | A10G (24GB) | 24GB | 7B models (FP16) | $1.01 |
| g5.2xlarge | A10G (24GB) | 24GB | 7B-13B models | $1.21 |
| g5.12xlarge | 4× A10G | 96GB | 70B models | $5.67 |
| p3.2xlarge | V100 (16GB) | 16GB | 7B models (budget) | $3.06 |
| p4d.24xlarge | 8× A100 (40GB) | 320GB | 70B+ models | $32.77 |
| p5.48xlarge | 8× H100 (80GB) | 640GB | 405B models | ~$40 |

**Recommended for most users:** `g5.xlarge` (A10G 24GB) - great price/performance for 7B models

### Step 2: Launch EC2 Instance (AWS Console)

**From your Mac browser:**

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Name your instance:**
   ```
   Name: llm-inference-server
   ```

3. **Choose AMI (Amazon Machine Image):**
   - Select: **Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04)**
   - This includes: NVIDIA drivers, CUDA, Docker, nvidia-container-toolkit

4. **Choose Instance Type:**
   - Select: **g5.xlarge** (1× A10G 24GB)

5. **Key Pair (for SSH access):**
   - Create new key pair: `llm-server-key`
   - Download `llm-server-key.pem` to your Mac
   - **Important:** Save this file, you need it to SSH

6. **Network Settings:**
   - Create security group with these rules:
     - SSH (port 22): Your IP only
     - Custom TCP (port 8000): Your IP only (for vLLM API)
     - Custom TCP (port 8001): Your IP only (for orchestrator)

7. **Storage:**
   - Configure: **100 GB gp3** (for model storage)

8. **Launch Instance**

### Step 3: Connect from Your Mac

**Wait 2-3 minutes for instance to start**, then:

```bash
# On your Mac terminal

# 1. Set permissions on your key file
chmod 400 ~/Downloads/llm-server-key.pem

# 2. Get your instance public IP from AWS Console
# Look for "Public IPv4 address" (e.g., 54.123.45.67)

# 3. SSH into your instance
ssh -i ~/Downloads/llm-server-key.pem ubuntu@54.123.45.67
```

**You should now be connected to your cloud GPU instance!**

### Step 4: Verify GPU Access

```bash
# Check GPU is available
nvidia-smi

# You should see:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
# |  0%   25C    P0    53W / 300W |      0MiB / 23028MiB |      0%      Default |
```

### Step 5: Verify Docker is Ready

```bash
# Docker should already be installed on Deep Learning AMI
docker --version
# Docker version 24.0.x

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi

# You should see the same GPU info as above
```

**Your AWS EC2 instance is now ready for LLM deployment!**

---

## GCP Compute Engine GPU Setup {#gcp-compute}

### Step 1: Choose Your Instance Type

| Instance Type | GPU | VRAM | Use Case | Cost/Hour |
|---------------|-----|------|----------|-----------|
| n1-standard-4 + T4 | T4 (16GB) | 16GB | 7B models (budget) | $0.35 + $0.35 = $0.70 |
| n1-standard-8 + L4 | L4 (24GB) | 24GB | 7B-13B models | $0.38 + $0.80 = $1.18 |
| a2-highgpu-1g | A100 (40GB) | 40GB | 70B models | $3.67 |
| a2-ultragpu-1g | A100 (80GB) | 80GB | 70B models | $5.51 |

**Recommended:** `n1-standard-8 + L4` (24GB) - best price/performance

### Step 2: Launch Compute Engine Instance (GCP Console)

**From your Mac browser:**

1. **Go to GCP Console** → Compute Engine → VM Instances → Create Instance

2. **Configure Instance:**
   ```
   Name: llm-inference-server
   Region: us-central1 (Iowa) - cheapest GPU region
   Zone: us-central1-a
   ```

3. **Machine Configuration:**
   - Series: **N1**
   - Machine type: **n1-standard-8** (8 vCPU, 30 GB memory)

4. **GPU:**
   - Click "Add GPU"
   - GPU type: **NVIDIA L4**
   - Number of GPUs: **1**

5. **Boot Disk:**
   - Click "Change"
   - Operating System: **Ubuntu**
   - Version: **Ubuntu 22.04 LTS**
   - Boot disk type: **Balanced persistent disk**
   - Size: **100 GB**

6. **Firewall:**
   - ✅ Allow HTTP traffic
   - ✅ Allow HTTPS traffic

7. **Advanced → Networking:**
   - Add network tag: `llm-server`

8. **Create**

### Step 3: Set Up Firewall Rules

**In GCP Console:**

1. **VPC Network** → Firewall → Create Firewall Rule

2. **Rule 1: SSH Access**
   ```
   Name: allow-ssh-llm
   Targets: Specified target tags → llm-server
   Source IP ranges: Your IP (find at https://whatismyip.com)
   Protocols and ports: tcp:22
   ```

3. **Rule 2: vLLM API Access**
   ```
   Name: allow-vllm-api
   Targets: Specified target tags → llm-server
   Source IP ranges: Your IP
   Protocols and ports: tcp:8000
   ```

### Step 4: Connect from Your Mac

```bash
# On your Mac terminal

# 1. Install gcloud CLI (if not already installed)
brew install google-cloud-sdk

# 2. Authenticate
gcloud auth login

# 3. Set your project
gcloud config set project YOUR_PROJECT_ID

# 4. SSH into your instance
gcloud compute ssh llm-inference-server --zone=us-central1-a
```

### Step 5: Install NVIDIA Drivers and Docker

```bash
# Once connected to your GCP instance

# 1. Install NVIDIA drivers
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Reboot to load drivers
sudo reboot

# 5. Wait 1 minute, then reconnect
gcloud compute ssh llm-inference-server --zone=us-central1-a

# 6. Verify GPU
nvidia-smi

# 7. Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Your GCP instance is now ready!**

---

## Azure GPU VM Setup {#azure-vm}

### Step 1: Choose Your VM Size

| VM Size | GPU | VRAM | Use Case | Cost/Hour |
|---------|-----|------|----------|-----------|
| NC6s_v3 | V100 (16GB) | 16GB | 7B models | $3.06 |
| NC24ads_A100_v4 | A100 (80GB) | 80GB | 70B models | $3.67 |
| ND96asr_v4 | 8× A100 (40GB) | 320GB | 70B+ models | $27.20 |

### Step 2: Launch VM (Azure Portal)

**From your Mac browser:**

1. **Go to Azure Portal** → Virtual Machines → Create

2. **Basics:**
   ```
   Subscription: Your subscription
   Resource Group: Create new → "llm-rg"
   VM Name: llm-inference-server
   Region: East US (cheapest for GPU)
   Image: Ubuntu Server 22.04 LTS - Gen2
   Size: NC6s_v3 (1× V100 16GB)
   ```

3. **Administrator Account:**
   ```
   Authentication type: SSH public key
   Username: azureuser
   SSH public key source: Generate new key pair
   Key pair name: llm-server-key
   ```
   - **Download the private key** when prompted

4. **Inbound Port Rules:**
   - Select: SSH (22)

5. **Disks:**
   - OS disk size: **100 GB Premium SSD**

6. **Review + Create**

### Step 3: Add Firewall Rules for API Access

1. **VM** → Networking → Add inbound port rule
   ```
   Source: My IP address
   Destination port ranges: 8000
   Protocol: TCP
   Name: allow-vllm-api
   ```

### Step 4: Connect from Your Mac

```bash
# On your Mac terminal

# 1. Set permissions on downloaded key
chmod 400 ~/Downloads/llm-server-key.pem

# 2. Get your VM public IP from Azure Portal
# Look for "Public IP address"

# 3. SSH into your VM
ssh -i ~/Downloads/llm-server-key.pem azureuser@20.123.45.67
```

### Step 5: Install NVIDIA Drivers and Docker

```bash
# Once connected to your Azure VM

# 1. Install NVIDIA drivers (Azure-specific)
sudo apt-get update
sudo apt-get install -y linux-azure
sudo apt-get install -y nvidia-driver-535 nvidia-utils-535

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Reboot
sudo reboot

# 5. Wait 1 minute, reconnect, verify
ssh -i ~/Downloads/llm-server-key.pem azureuser@20.123.45.67
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Your Azure VM is now ready!**

---

## Deploying vLLM Container {#deploy-vllm}

**Run these commands on your cloud GPU instance (after SSH-ing in)**

### Option 1: Quick Start with Hugging Face Model

```bash
# Deploy Qwen 7B model with vLLM (pulls model from Hugging Face)
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# Check logs
docker logs -f vllm-server

# Wait for "Application startup complete" message
```

### Option 2: Upload Your Own Model from Mac

```bash
# On your Mac (local terminal)

# 1. Upload your model directory to cloud instance
# Replace with your actual model path and instance IP
scp -i ~/Downloads/llm-server-key.pem -r \
  ./my_fine_tuned_model \
  ubuntu@54.123.45.67:~/models/

# On your cloud instance (SSH terminal)

# 2. Deploy your model
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -v ~/models:/models \
  vllm/vllm-openai:latest \
  --model /models/my_fine_tuned_model \
  --dtype float16 \
  --gpu-memory-utilization 0.9
```

### Test the API from Your Mac

```bash
# On your Mac terminal

# Replace with your instance public IP
export INSTANCE_IP=54.123.45.67

# Test the API
curl http://${INSTANCE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 50
  }'
```

### Python Client from Your Mac

```python
# On your Mac, create test_client.py
from openai import OpenAI

# Replace with your instance IP
client = OpenAI(
    base_url="http://54.123.45.67:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about cloud computing"}
    ]
)

print(response.choices[0].message.content)
```

```bash
# Run from your Mac
python test_client.py
```

---

## Deploying llama.cpp Container {#deploy-llama-cpp}

**For GGUF quantized models (smaller memory footprint)**

### Step 1: Upload GGUF Model from Mac

```bash
# On your Mac

# Download a GGUF model (example: Qwen 1.5B Q8)
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf

# Upload to cloud instance
scp -i ~/Downloads/llm-server-key.pem \
  qwen2.5-1.5b-instruct-q8_0.gguf \
  ubuntu@54.123.45.67:~/models/
```

### Step 2: Run llama.cpp Server on Cloud Instance

```bash
# On your cloud instance (SSH terminal)

# Create models directory
mkdir -p ~/models

# Run llama.cpp server with GPU acceleration
docker run -d \
  --name llama-server \
  --gpus all \
  -p 8080:8080 \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 4096

# Check logs
docker logs -f llama-server
```

### Step 3: Test from Your Mac

```bash
# On your Mac

export INSTANCE_IP=54.123.45.67

# Test completion
curl http://${INSTANCE_IP}:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## Cost Comparison & Recommendations {#cost-comparison}

### Monthly Cost Estimates (24/7 Operation)

| Provider | Instance | GPU | Cost/Hour | Cost/Month (730h) | Best For |
|----------|----------|-----|-----------|-------------------|----------|
| AWS | g5.xlarge | A10G 24GB | $1.01 | $737 | Production 7B |
| AWS | g5.12xlarge | 4× A10G | $5.67 | $4,139 | Production 70B |
| GCP | n1-std-8 + L4 | L4 24GB | $1.18 | $861 | Development |
| GCP | a2-highgpu-1g | A100 40GB | $3.67 | $2,679 | Production 70B |
| Azure | NC6s_v3 | V100 16GB | $3.06 | $2,234 | Budget option |

### Cost Optimization Strategies

**1. Use Spot/Preemptible Instances (70% discount)**

```bash
# AWS Spot Instance (via console)
# Select: "Request Spot Instances" when launching
# Typical cost: $0.30/hour (vs $1.01 regular)

# GCP Preemptible (via console)
# Select: "Preemptible VM instance" checkbox
# Typical cost: $0.35/hour (vs $1.18 regular)
```

**2. Auto-Stop When Idle**

```bash
# On your cloud instance
# Create auto-stop script: ~/auto-stop.sh

#!/bin/bash
# Stop instance if no requests for 1 hour

IDLE_TIME=3600  # 1 hour in seconds
LAST_REQUEST=$(docker logs vllm-server | grep "POST /v1" | tail -1 | cut -d' ' -f1)
CURRENT_TIME=$(date +%s)

if [ $((CURRENT_TIME - LAST_REQUEST)) -gt $IDLE_TIME ]; then
  sudo shutdown -h now
fi

# Add to crontab (runs every 15 minutes)
crontab -e
# Add: */15 * * * * /home/ubuntu/auto-stop.sh
```

**3. Scale Down During Off-Hours**

```bash
# AWS CLI (from your Mac)
# Stop instance at night
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start in the morning
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### Recommendations by Use Case

| Use Case | Recommended Setup | Monthly Cost |
|----------|-------------------|--------------|
| Development/Testing | GCP n1-std-4 + T4 (preemptible) | ~$150 |
| Staging (7B model) | AWS g5.xlarge (spot) | ~$220 |
| Production (7B, 24/7) | AWS g5.xlarge | $737 |
| Production (70B, 24/7) | AWS g5.12xlarge | $4,139 |
| Production (70B, business hours only) | AWS g5.12xlarge (12h/day) | ~$2,000 |

---

## Mac Local Testing (CPU-Only) {#mac-local-testing}

**For quick testing without spinning up cloud instances**

### Option 1: llama.cpp Native (Best Performance on Mac)

```bash
# On your Mac

# Install llama.cpp
brew install llama.cpp

# Download a small GGUF model
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Run inference (uses Mac CPU/Metal acceleration)
llama-server \
  -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -ngl 1 \
  -c 2048 \
  --host 127.0.0.1 \
  --port 8080

# Test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Performance expectations on Mac:**
- M1/M2 Mac: 5-15 tokens/second (1.5B-3B models)
- M3 Max: 10-25 tokens/second (1.5B-7B models)
- **Not suitable for production, only for code testing**

### Option 2: Ollama (Easiest for Mac)

```bash
# On your Mac

# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull and run a model
ollama run qwen2.5:1.5b

# API is now available at http://localhost:11434
```

**When to use Mac local testing:**
- Testing API client code before deploying to cloud
- Rapid prototyping with small models
- Demos without internet connection
- **Never for production workloads**

---

## Deployment Workflow Summary

### Development Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DEVELOP ON MAC                                               │
│    - Test API clients                                           │
│    - Use Ollama/llama.cpp for quick iterations                  │
│    - Write Docker Compose configurations                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DEPLOY TO SINGLE CLOUD GPU (Staging)                        │
│    - SSH into AWS/GCP/Azure instance                            │
│    - Run vLLM/llama.cpp Docker container                        │
│    - Test with real GPU performance                             │
│    - Measure latency, throughput, costs                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. SCALE TO PRODUCTION (Kubernetes/Multi-GPU)                  │
│    - See ENTERPRISE_SCALE_DEPLOYMENT.md                         │
│    - Multi-replica deployment                                   │
│    - Autoscaling, monitoring, load balancing                    │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

**From your Mac, deploy to AWS in 5 minutes:**

```bash
# 1. Launch g5.xlarge instance in AWS Console
# 2. SSH in
ssh -i llm-server-key.pem ubuntu@YOUR_INSTANCE_IP

# 3. Run vLLM (already on the instance)
docker run -d --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct

# 4. Test from Mac
curl http://YOUR_INSTANCE_IP:8000/v1/models
```

---

## Troubleshooting

### "nvidia-smi: command not found"

```bash
# Install NVIDIA drivers (Ubuntu)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### "could not select device driver with capabilities: [[gpu]]"

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### "Connection refused" when testing API from Mac

```bash
# Check firewall allows your IP on port 8000
# AWS: Security Groups
# GCP: Firewall Rules
# Azure: Network Security Group

# Verify server is running
docker ps
docker logs vllm-server
```

### Model download is too slow

```bash
# Use Hugging Face mirror (if in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download model on Mac, then upload
# Download to Mac:
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen-7b

# Upload to cloud:
scp -i key.pem -r ./qwen-7b ubuntu@INSTANCE_IP:~/models/
```

---

## Next Steps

Once your single GPU instance is working:

1. **Add monitoring**: Install Prometheus + Grafana for metrics
2. **Set up CI/CD**: Automate deployments with GitHub Actions
3. **Scale horizontally**: Move to Kubernetes (see ENTERPRISE_SCALE_DEPLOYMENT.md)
4. **Add load balancing**: Use cloud load balancers for multiple instances
5. **Optimize costs**: Use spot instances, auto-scaling, scheduled shutdown

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Maintained for:** Mac developers deploying to cloud GPU infrastructure
