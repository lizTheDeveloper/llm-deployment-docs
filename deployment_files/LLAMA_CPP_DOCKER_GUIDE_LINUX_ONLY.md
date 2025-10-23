# LLaMA.cpp Docker Setup Guide (Linux with NVIDIA GPU Only)

> **⚠️ WARNING FOR MAC USERS:**
>
> **This guide does NOT work on Mac!** Macs cannot use NVIDIA GPUs with Docker.
>
> **Mac users: Use [CLOUD_GPU_DEPLOYMENT_GUIDE.md](CLOUD_GPU_DEPLOYMENT_GUIDE.md) instead.**
>
> That guide shows you how to:
> - Deploy to AWS/GCP/Azure GPU instances from your Mac
> - Test locally on Mac with CPU-only inference (Ollama/llama.cpp native)
> - Actual step-by-step instructions that work for Mac developers
>
> This guide is only for Linux users with NVIDIA GPUs running locally.

---

Complete guide for running LLaMA.cpp in Docker containers with GPU acceleration on **Linux systems with NVIDIA GPUs only**.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Command-Line Setup](#command-line-setup)
3. [GPU-Accelerated Setup](#gpu-accelerated-setup)
4. [Server Mode Setup](#server-mode-setup)
5. [Docker Desktop UI Instructions](#docker-desktop-ui-instructions)

---

## Prerequisites

### Required Software
- **Docker Desktop** (or Docker Engine + Docker CLI)
  - Download: https://www.docker.com/products/docker-desktop
  - Verify installation: `docker --version`
- **NVIDIA Container Toolkit** (for GPU support)
  - Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Create Model Directory
```bash
# Create directory for storing models
mkdir -p ~/llama-models
cd ~/llama-models
```

---

## Command-Line Setup

### Step 1: Download a GGUF Model

Download a pre-quantized model from Hugging Face:

```bash
# Option 1: Small model (1.5B parameters, ~1GB)
wget https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -O ~/llama-models/qwen-1.5b-q8.gguf

# Option 2: Medium model (7B parameters, ~4GB)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
  -O ~/llama-models/llama-2-7b-q4.gguf

# Option 3: Larger model (13B parameters, ~8GB)
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf \
  -O ~/llama-models/llama-2-13b-q4.gguf
```

**Tip:** Use `Q4_K_M` or `Q8_0` quantization for best quality-to-size ratio.

### Step 2: Run Inference (CPU)

**Single prompt inference:**
```bash
docker run -it --rm \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:light \
  -m /models/qwen-1.5b-q8.gguf \
  -p "Explain quantum computing in simple terms:" \
  -n 256 \
  --temp 0.7
```

**Interactive chat mode:**
```bash
docker run -it --rm \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:light \
  -m /models/qwen-1.5b-q8.gguf \
  --interactive-first \
  --color \
  -r "User:" \
  --in-prefix " " \
  -n 512
```

**Parameters explained:**
- `-m /models/model.gguf` - Model file path
- `-p "prompt"` - Initial prompt
- `-n 256` - Maximum tokens to generate
- `--temp 0.7` - Temperature (0.0 = deterministic, 1.0 = creative)
- `--interactive-first` - Enable chat mode
- `--color` - Enable colored output
- `-r "User:"` - Reverse prompt to stop generation

### Step 3: Advanced Options

**Adjust context size:**
```bash
docker run -it --rm \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:light \
  -m /models/llama-2-7b-q4.gguf \
  -p "Write a story about:" \
  -n 512 \
  -c 4096 \
  --temp 0.8
```
- `-c 4096` - Context window size (max input + output tokens)

**Control output formatting:**
```bash
docker run -it --rm \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:light \
  -m /models/qwen-1.5b-q8.gguf \
  -p "List 5 benefits of exercise:" \
  -n 256 \
  --repeat-penalty 1.1 \
  --top-k 40 \
  --top-p 0.9
```
- `--repeat-penalty 1.1` - Reduce repetition (1.0 = none, 1.5 = strong)
- `--top-k 40` - Sample from top K tokens
- `--top-p 0.9` - Nucleus sampling threshold

---

## GPU-Accelerated Setup

### Step 1: Verify NVIDIA Setup

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If these commands fail, install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Build GPU-Enabled Image

```bash
# Clone llama.cpp repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build CUDA-enabled image (full version with tools)
docker build -t llama-cpp:cuda-full \
  --target full \
  -f .devops/cuda.Dockerfile .

# Build CUDA-enabled server (for API usage)
docker build -t llama-cpp:cuda-server \
  --target server \
  -f .devops/cuda.Dockerfile .
```

**Build for specific GPU architecture (optional):**
```bash
# For RTX 3000/4000 series (Ampere/Ada)
docker build -t llama-cpp:cuda-full \
  --build-arg CUDA_DOCKER_ARCH="compute_86;compute_89" \
  --target full \
  -f .devops/cuda.Dockerfile .

# For RTX 2000 series (Turing)
docker build -t llama-cpp:cuda-full \
  --build-arg CUDA_DOCKER_ARCH="compute_75" \
  --target full \
  -f .devops/cuda.Dockerfile .
```

### Step 3: Run with GPU Acceleration

```bash
# Run inference with GPU (offload all layers)
docker run -it --rm \
  --gpus all \
  -v ~/llama-models:/models \
  llama-cpp:cuda-full \
  --run \
  -m /models/llama-2-7b-q4.gguf \
  -p "Explain neural networks:" \
  -n 512 \
  --n-gpu-layers 99 \
  --temp 0.7
```

**Parameters for GPU:**
- `--gpus all` - Use all available GPUs
- `--n-gpu-layers 99` - Offload layers to GPU (99 = all layers)
- `--gpus '"device=0"'` - Use specific GPU (for multi-GPU systems)

**Multi-GPU setup:**
```bash
# Use GPUs 0 and 1
docker run -it --rm \
  --gpus '"device=0,1"' \
  -v ~/llama-models:/models \
  llama-cpp:cuda-full \
  --run \
  -m /models/llama-2-13b-q4.gguf \
  -p "What is machine learning?" \
  -n 256 \
  --n-gpu-layers 99 \
  --tensor-split 3,1
```
- `--tensor-split 3,1` - Split model 75%/25% across GPUs

### Step 4: Monitor GPU Usage

```bash
# In another terminal, watch GPU utilization
watch -n 1 nvidia-smi
```

---

## Server Mode Setup

### Step 1: Start LLaMA.cpp Server

**CPU Server:**
```bash
docker run -d \
  --name llama-server \
  -p 8080:8080 \
  -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen-1.5b-q8.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -n 512
```

**GPU Server:**
```bash
docker run -d \
  --name llama-server-gpu \
  --gpus all \
  -p 8080:8080 \
  -v ~/llama-models:/models \
  llama-cpp:cuda-server \
  -m /models/llama-2-7b-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -n 512 \
  --n-gpu-layers 99
```

**Parameters:**
- `-d` - Run in background (detached mode)
- `--name llama-server` - Container name
- `-p 8080:8080` - Expose port 8080
- `--host 0.0.0.0` - Listen on all interfaces

### Step 2: Test Server with curl

**Completion API:**
```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Building a website can be done in 10 simple steps:",
    "n_predict": 128,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
  }'
```

**Chat API (OpenAI-compatible):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain Docker in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**Streaming response:**
```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a short story:",
    "n_predict": 256,
    "stream": true
  }' \
  --no-buffer
```

### Step 3: Use with OpenAI Python SDK

```python
from openai import OpenAI

# Point to local llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="not-used",  # Model is already loaded in server
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is quantum computing?"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

### Step 4: Server Management

```bash
# View server logs
docker logs -f llama-server

# Stop server
docker stop llama-server

# Restart server
docker restart llama-server

# Remove server container
docker rm llama-server

# Check server status
curl http://localhost:8080/health
```

---

## Docker Desktop UI Instructions

### Initial Setup

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Run the installer
   - Start Docker Desktop application
   - Wait for "Docker Desktop is running" in system tray

2. **Enable GPU Support (Windows with WSL2)**
   - Open Docker Desktop
   - Go to **Settings** → **Resources** → **WSL Integration**
   - Enable integration with your WSL distribution
   - Click **Apply & Restart**

3. **Verify Installation**
   - Open Docker Desktop
   - Go to **Settings** → **General**
   - Ensure "Use the WSL 2 based engine" is checked (Windows)
   - Click **Apply & Restart** if needed

### Running LLaMA.cpp via UI

#### Method 1: Using Docker Desktop Images Tab

1. **Pull the Image**
   - Open Docker Desktop
   - Click **Images** in left sidebar
   - Click **Pull** button (top right)
   - Enter: `ghcr.io/ggml-org/llama.cpp:light`
   - Click **Pull**
   - Wait for download to complete

2. **Prepare Model File**
   - Create folder on your computer: `C:\llama-models` (Windows) or `~/llama-models` (Mac/Linux)
   - Download a GGUF model file into this folder
   - Example: Download from https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF

3. **Run Container**
   - In **Images** tab, find `ghcr.io/ggml-org/llama.cpp:light`
   - Click the **▶️ Run** button
   - Click **Optional settings** dropdown
   - Configure:
     - **Container name**: `llama-inference`
     - **Volumes** → Click **+** button:
       - **Host path**: `C:\llama-models` (or `~/llama-models`)
       - **Container path**: `/models`
     - **Command**:
       ```
       -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf -p "Hello, how are you?" -n 100
       ```
   - Click **Run**

4. **View Output**
   - Container will appear in **Containers** tab
   - Click on the container name
   - Go to **Logs** tab to see the generated text

#### Method 2: Running Server Mode via UI

1. **Pull Server Image**
   - **Images** → **Pull**
   - Enter: `ghcr.io/ggml-org/llama.cpp:server`
   - Click **Pull**

2. **Run Server Container**
   - Find the server image in **Images** tab
   - Click **▶️ Run**
   - Click **Optional settings**
   - Configure:
     - **Container name**: `llama-server`
     - **Ports**:
       - **Host port**: `8080`
       - **Container port**: `8080`
     - **Volumes**:
       - **Host path**: `C:\llama-models`
       - **Container path**: `/models`
     - **Command**:
       ```
       -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf --host 0.0.0.0 --port 8080
       ```
   - Click **Run**

3. **Verify Server is Running**
   - Go to **Containers** tab
   - Container status should show **Running**
   - Click container → **Logs** tab to see startup messages
   - Open browser to: http://localhost:8080

4. **Test API from Browser**
   - Install a REST client browser extension (e.g., Postman, Thunder Client)
   - Send POST request to: `http://localhost:8080/completion`
   - Body (JSON):
     ```json
     {
       "prompt": "What is Docker?",
       "n_predict": 100
     }
     ```

#### Method 3: Using Docker Compose (Advanced)

1. **Create Docker Compose File**
   - Create file: `docker-compose.yml` in a folder
   - Contents:
     ```yaml
     version: '3.8'
     services:
       llama-server:
         image: ghcr.io/ggml-org/llama.cpp:server
         ports:
           - "8080:8080"
         volumes:
           - ~/llama-models:/models
         command: >
           -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
           --host 0.0.0.0
           --port 8080
           -c 4096
     ```

2. **Deploy via Docker Desktop**
   - Open Docker Desktop
   - Click **Containers** → **⋮** (three dots) → **Deploy from Compose file**
   - Select your `docker-compose.yml` file
   - Click **Deploy**

3. **Manage Stack**
   - Find your stack in **Containers** tab
   - Expand to see running services
   - Click **⏹️** to stop, **▶️** to start
   - Click **🗑️** to remove entire stack

### GPU-Enabled Containers via UI (NVIDIA Only)

1. **Verify GPU Access**
   - Ensure NVIDIA Container Toolkit is installed
   - Docker Desktop should auto-detect GPU

2. **Build Custom GPU Image**
   - Clone llama.cpp: `git clone https://github.com/ggml-org/llama.cpp.git`
   - Open Docker Desktop
   - Go to **Images** → **Build**
   - Navigate to `llama.cpp/.devops/cuda.Dockerfile`
   - Click **Build**
   - Tag as: `llama-cpp:cuda`

3. **Run GPU Container**
   - Find `llama-cpp:cuda` in **Images**
   - Click **Run**
   - Click **Optional settings**
   - Add **Environment variable**:
     - Name: `NVIDIA_VISIBLE_DEVICES`
     - Value: `all`
   - Configure volumes, ports as before
   - Add to **Command**:
     ```
     -m /models/llama-2-7b-q4.gguf -p "Hello" --n-gpu-layers 99
     ```

### Monitoring and Troubleshooting via UI

1. **View Container Logs**
   - **Containers** → Click container name
   - **Logs** tab shows real-time output
   - Use search box to filter logs

2. **Inspect Container**
   - Click container → **Inspect** tab
   - View configuration, environment, volumes
   - Copy useful details for debugging

3. **Execute Commands Inside Container**
   - Click container → **Exec** tab
   - Enter commands like `ls /models` to verify files
   - Or click **Terminal** to open interactive shell

4. **Resource Usage**
   - Click container → **Stats** tab
   - Monitor CPU, memory, network usage in real-time
   - Useful for identifying performance bottlenecks

5. **Common Issues**
   - **Model not found**: Check volume path in **Inspect** → **Mounts**
   - **Port already in use**: Change host port in settings
   - **Container exits immediately**: Check **Logs** for error messages
   - **GPU not detected**: Verify NVIDIA toolkit with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

---

## Best Practices

### Performance Tips

1. **Choose the right quantization:**
   - `Q4_K_M` - Best balance (4-bit, medium quality)
   - `Q5_K_M` - Better quality (5-bit)
   - `Q8_0` - Highest quality (8-bit)
   - `Q2_K` - Smallest size (2-bit, lower quality)

2. **Optimize GPU layers:**
   - Start with `--n-gpu-layers 99` (all layers)
   - If running out of VRAM, reduce to 40-50
   - Monitor with `nvidia-smi` to find optimal setting

3. **Context size:**
   - Larger context = more VRAM needed
   - Start with `-c 2048` for 7B models
   - Increase to `-c 4096` or `-c 8192` if needed

### Security Tips

1. **Don't expose server to internet without authentication**
   - Add reverse proxy (nginx) with auth
   - Use VPN or SSH tunnels for remote access

2. **Limit resource usage:**
   ```bash
   docker run --memory="8g" --cpus="4" ...
   ```

3. **Use read-only volumes for models:**
   ```bash
   -v ~/llama-models:/models:ro
   ```

### Troubleshooting

**Container exits immediately:**
```bash
# Check logs
docker logs llama-server

# Run in foreground to see errors
docker run -it --rm ... (remove -d flag)
```

**Out of memory:**
```bash
# Reduce context size
-c 1024

# Use smaller model or quantization
# Q4_K_M instead of Q8_0

# Reduce GPU layers
--n-gpu-layers 32
```

**Slow inference:**
```bash
# Enable GPU acceleration
--n-gpu-layers 99

# Reduce batch size
-b 512

# Use optimized model format (GGUF, not GGML)
```

---

## Resources

- **Official llama.cpp repo**: https://github.com/ggml-org/llama.cpp
- **GGUF model repository**: https://huggingface.co/models?library=gguf
- **Docker documentation**: https://docs.docker.com/
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

## Quick Reference

### Essential Commands

```bash
# Pull latest image
docker pull ghcr.io/ggml-org/llama.cpp:light

# Run single inference
docker run -it --rm -v ~/llama-models:/models ghcr.io/ggml-org/llama.cpp:light \
  -m /models/model.gguf -p "prompt" -n 256

# Start server
docker run -d --name llama-server -p 8080:8080 -v ~/llama-models:/models \
  ghcr.io/ggml-org/llama.cpp:server -m /models/model.gguf --host 0.0.0.0 --port 8080

# GPU inference
docker run -it --rm --gpus all -v ~/llama-models:/models llama-cpp:cuda-full \
  --run -m /models/model.gguf -p "prompt" -n 256 --n-gpu-layers 99

# Stop all llama containers
docker stop $(docker ps -q --filter ancestor=ghcr.io/ggml-org/llama.cpp:server)
```

### Useful Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-m` | Model path | `-m /models/model.gguf` |
| `-p` | Prompt text | `-p "Hello world"` |
| `-n` | Max tokens | `-n 256` |
| `-c` | Context size | `-c 4096` |
| `--temp` | Temperature | `--temp 0.7` |
| `--n-gpu-layers` | GPU layers | `--n-gpu-layers 99` |
| `--repeat-penalty` | Repetition penalty | `--repeat-penalty 1.1` |
| `--top-k` | Top-K sampling | `--top-k 40` |
| `--top-p` | Nucleus sampling | `--top-p 0.9` |
| `--interactive-first` | Chat mode | `--interactive-first` |
