# How to Deploy Lab 8 & 9 to AWS ECS (Production-Grade with vLLM)

This guide walks you through deploying Lab 8 (OpenAI-Compatible API) and Lab 9 (with Tool Calling) as production-grade containerized services on AWS Elastic Container Service (ECS) using **vLLM**, the industry-standard high-performance inference engine.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Lab 8: Basic OpenAI-Compatible API Deployment](#lab-8-basic-openai-compatible-api-deployment)
4. [Lab 9: Tool Calling API Deployment](#lab-9-tool-calling-api-deployment)
5. [AWS Setup](#aws-setup)
6. [Building and Pushing Docker Images](#building-and-pushing-docker-images)
7. [ECS Deployment](#ecs-deployment)
8. [Testing the Deployment](#testing-the-deployment)
9. [Monitoring and Scaling](#monitoring-and-scaling)
10. [Cleanup](#cleanup)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Requirements
- Docker Desktop installed and running
- AWS CLI v2 installed and configured (`aws configure`)
- Python 3.10 or higher
- Git

### AWS Requirements
- AWS Account with appropriate permissions
- IAM user with permissions for:
  - ECR (Elastic Container Registry)
  - ECS (Elastic Container Service)
  - EC2 (for networking and load balancing)
  - CloudWatch (for logging)
  - IAM (for task execution roles)

### Recommended AWS Instance Type
- **Single GPU**: `g4dn.xlarge` (1x T4, 16GB VRAM) - good for 7B models
- **Multi-GPU**: `g4dn.12xlarge` (4x T4, 64GB total VRAM) - for larger models or tensor parallelism
- **High Performance**: `p3.2xlarge` (1x V100, 16GB VRAM) or `g5.xlarge` (1x A10G, 24GB VRAM)
- **Production**: `g5.2xlarge` or larger for better throughput
- Note: vLLM requires GPU. CPU-only deployment is not recommended for production.

---

## Why vLLM?

vLLM is the production-standard LLM inference engine that provides:

- **PagedAttention**: Efficient KV cache management, reducing memory waste by up to 80%
- **Continuous Batching**: Dynamic batching of requests for maximum throughput
- **Optimized CUDA Kernels**: 2-10x faster than naive implementations
- **Native OpenAI API**: Built-in `/v1/chat/completions` endpoint - no custom code needed
- **Tool Calling Support**: Native support for function/tool calling (OpenAI format)
- **Tensor Parallelism**: Multi-GPU support out of the box
- **Streaming**: Built-in streaming response support
- **Production Ready**: Used by major companies for production LLM serving

## Project Structure

Create the following directory structure for each lab:

```
lab8_deployment/
├── Dockerfile
├── vllm_config.json
└── .dockerignore

lab9_deployment/
├── Dockerfile
├── vllm_config.json
├── tool_handler.py
└── .dockerignore
```

---

## Lab 8: Basic OpenAI-Compatible API Deployment with vLLM

### Step 1: Create Lab 8 Project Files

#### `lab8_deployment/vllm_config.json`

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "served_model_name": "llama-3.1-8b-instruct",
  "host": "0.0.0.0",
  "port": 8000,
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.90,
  "max_model_len": 4096,
  "dtype": "auto",
  "enable_chunked_prefill": true,
  "max_num_batched_tokens": 8192,
  "max_num_seqs": 256,
  "trust_remote_code": true,
  "disable_log_requests": false,
  "uvicorn_log_level": "info"
}
```

**Configuration Explained:**
- `gpu_memory_utilization`: 0.90 = use 90% of GPU memory for KV cache
- `max_model_len`: Maximum sequence length (context window)
- `enable_chunked_prefill`: Better throughput for long prompts
- `max_num_batched_tokens`: Total tokens across all sequences in a batch
- `max_num_seqs`: Maximum number of concurrent sequences

#### `lab8_deployment/Dockerfile`

```dockerfile
FROM vllm/vllm-openai:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Copy configuration
COPY vllm_config.json /app/vllm_config.json

# Expose port
EXPOSE 8000

# Health check - vLLM has built-in /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run vLLM server with configuration from JSON
# We'll use environment variables to override as needed
CMD ["sh", "-c", "\
    python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL:-meta-llama/Llama-3.1-8B-Instruct} \
    --served-model-name ${SERVED_MODEL_NAME:-llama-3.1-8b-instruct} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.90} \
    --max-model-len ${MAX_MODEL_LEN:-4096} \
    --enable-chunked-prefill \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192} \
    --max-num-seqs ${MAX_NUM_SEQS:-256} \
    --trust-remote-code \
    --disable-log-stats=false \
    "]
```

#### `lab8_deployment/.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
*.md
.DS_Store
```

---

## Lab 9: Production Tool Calling API (Two-Tier Architecture)

### Architecture Overview

**Enterprise Pattern: Orchestration Layer + Inference Layer**

```
Client Request
    ↓
Orchestration Layer (port 8001, CPU)
    ├─→ vLLM Inference (port 8000, GPU) - generates tool call
    ├─→ Execute Tool (CRM API, database, external services, etc.)
    ├─→ vLLM Inference (with tool result) - generates final response
    └─→ Return to client
```

**Why Two Tiers?**
- **Separation of Concerns**: Business logic vs pure inference
- **Independent Scaling**: Scale orchestrator on cheap CPU, vLLM on expensive GPU
- **Cost Optimization**: GPU only used for inference (~100ms), not waiting for API calls (seconds)
- **Multi-Tenancy**: Different tool sets per customer/org, isolated execution
- **Flexibility**: Deploy tool updates without redeploying inference layer
- **Security**: API keys, credentials, and tenant data stay in orchestration layer
- **Observability**: Track tool execution, latency, and success rates separately

**This is the production pattern used by major LLM platforms serving millions of users.**

### Architecture Decision Matrix

| Scale | Orchestration Layer | When to Use |
|-------|-------------------|-------------|
| **Learning/Prototyping** | FastAPI (simple) | <100 QPS, single-tenant |
| **Production (<1k QPS)** | FastAPI + Redis cache | Multi-tenant, moderate scale |
| **Enterprise Scale** | Ray Serve | >1k QPS, multi-region, A/B testing |
| **Ultra-low Latency** | Rust (Axum) + FastAPI | <50ms SLA requirements |

For this lab, we'll implement **FastAPI** (easier to understand), but include Ray Serve configuration for enterprise deployment.

### Step 1: Create Lab 9 Project Files

#### Directory Structure

```
lab9_deployment/
├── docker-compose.yml          # Local testing
├── vllm/
│   └── Dockerfile             # vLLM inference service
├── orchestrator/
│   ├── Dockerfile             # FastAPI orchestrator
│   ├── main.py               # Orchestration logic
│   ├── tools.py              # Tool implementations
│   └── requirements.txt      # Python dependencies
└── .dockerignore
```

#### `lab9_deployment/docker-compose.yml`

```yaml
version: '3.8'

services:
  # vLLM Inference Service (GPU)
  vllm:
    build: ./vllm
    container_name: vllm-inference
    ports:
      - "8000:8000"
    environment:
      - MODEL=meta-llama/Llama-3.1-8B-Instruct
      - TENSOR_PARALLEL_SIZE=1
      - GPU_MEMORY_UTILIZATION=0.90
      - MAX_MODEL_LEN=4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s

  # FastAPI Orchestrator (CPU)
  orchestrator:
    build: ./orchestrator
    container_name: fastapi-orchestrator
    ports:
      - "8001:8001"
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
      - WEATHER_API_KEY=${WEATHER_API_KEY:-demo-key}
    depends_on:
      vllm:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 15s
      timeout: 5s
      retries: 3
```

#### `lab9_deployment/vllm/Dockerfile`

```dockerfile
FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run vLLM server (tool calling enabled via model support)
CMD ["sh", "-c", "\
    python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL:-meta-llama/Llama-3.1-8B-Instruct} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.90} \
    --max-model-len ${MAX_MODEL_LEN:-4096} \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --disable-log-stats=false \
    "]
```

#### `lab9_deployment/orchestrator/tools.py`

```python
"""
Tool implementations for the orchestrator
These execute on CPU instances, not GPU
"""
from typing import Dict, Any
import random
import requests
import os


def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get current weather for a location
    In production, this would call a real weather API
    """
    # Simulate API call delay
    import time
    time.sleep(0.1)

    # Mock data (in production, call real API)
    # api_key = os.getenv("WEATHER_API_KEY")
    # response = requests.get(f"https://api.weather.com/v1/...", headers={...})

    temp = random.randint(15, 30)
    conditions = random.choice(["sunny", "cloudy", "rainy", "partly cloudy"])

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "conditions": conditions,
        "description": f"The weather in {location} is {conditions} with a temperature of {temp}°{unit[0].upper()}."
    }


def calculate_math(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression
    """
    try:
        # Security: only allow safe characters
        allowed = set("0123456789+-*/() .")
        if not all(c in allowed for c in expression):
            return {
                "success": False,
                "error": "Invalid characters in expression"
            }

        result = eval(expression)
        return {
            "success": True,
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e)
        }


def search_database(query: str) -> Dict[str, Any]:
    """
    Search internal database
    In production, this would query your actual database
    """
    # Simulate database query
    import time
    time.sleep(0.2)

    # Mock results
    results = [
        {"id": 1, "title": "Product A", "price": 99.99},
        {"id": 2, "title": "Product B", "price": 149.99}
    ]

    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


# Tool registry - easily extensible
AVAILABLE_TOOLS = {
    "get_current_weather": {
        "function": get_current_weather,
        "description": "Get the current weather for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    "calculate_math": {
        "function": calculate_math,
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '2+2' or '10*5'"
                }
            },
            "required": ["expression"]
        }
    },
    "search_database": {
        "function": search_database,
        "description": "Search the product database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments"""
    if tool_name not in AVAILABLE_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        tool_func = AVAILABLE_TOOLS[tool_name]["function"]
        return tool_func(**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def get_tool_definitions():
    """Get OpenAI-compatible tool definitions"""
    tools = []
    for name, config in AVAILABLE_TOOLS.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": config["description"],
                "parameters": config["parameters"]
            }
        })
    return tools
```

#### `lab9_deployment/orchestrator/main.py`

```python
"""
FastAPI Orchestrator for Tool Calling
Handles tool execution and multi-turn conversations with vLLM
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import os
import time
from tools import execute_tool, get_tool_definitions

app = FastAPI(title="Tool Calling Orchestrator")

# vLLM backend configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOOL_ROUNDS = 5  # Prevent infinite loops


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "orchestrator"}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    Orchestrate tool calling with vLLM
    This handles the full tool calling loop
    """
    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    tools = get_tool_definitions()

    tool_rounds = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        while tool_rounds < MAX_TOOL_ROUNDS:
            # Call vLLM for inference
            vllm_request = {
                "model": MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }

            try:
                response = await client.post(
                    f"{VLLM_BASE_URL}/chat/completions",
                    json=vllm_request
                )
                response.raise_for_status()
                result = response.json()

            except httpx.HTTPError as e:
                raise HTTPException(status_code=502, detail=f"vLLM backend error: {str(e)}")

            # Track token usage
            if "usage" in result:
                total_prompt_tokens += result["usage"].get("prompt_tokens", 0)
                total_completion_tokens += result["usage"].get("completion_tokens", 0)

            assistant_message = result["choices"][0]["message"]
            messages.append(assistant_message)

            # Check if model wants to call tools
            if not assistant_message.get("tool_calls"):
                # No tool calls - return final response
                return ChatResponse(
                    id=result["id"],
                    choices=result["choices"],
                    usage={
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens
                    },
                    metadata={
                        "tool_rounds": tool_rounds,
                        "backend": "vllm"
                    }
                )

            # Execute tools
            tool_rounds += 1
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                # Parse arguments if string
                if isinstance(tool_args, str):
                    import json
                    tool_args = json.loads(tool_args)

                # Execute tool (THIS HAPPENS ON ORCHESTRATOR, NOT vLLM!)
                tool_result = execute_tool(tool_name, tool_args)

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result)
                })

        # Max rounds reached
        raise HTTPException(
            status_code=500,
            detail=f"Maximum tool calling rounds ({MAX_TOOL_ROUNDS}) exceeded"
        )


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {"tools": get_tool_definitions()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### `lab9_deployment/orchestrator/requirements.txt`

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
httpx==0.25.0
```

#### `lab9_deployment/orchestrator/Dockerfile`

```dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY tools.py .

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### `lab9_deployment/.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
*.md
.DS_Store
```

### Step 2: Local Testing with Docker Compose

```bash
cd lab9_deployment

# Build and start both services
docker compose up --build

# In another terminal, test the orchestrator
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco and what is 15 times 23?"}
    ],
    "max_tokens": 500
  }'

# List available tools
curl http://localhost:8001/tools
```

---

## AWS Setup

### Step 1: Configure AWS CLI

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json)
```

### Step 2: Set Environment Variables

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REGISTRY=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### Step 3: Create ECR Repositories

```bash
# Create repository for Lab 8
aws ecr create-repository \
    --repository-name lab8-openai-api \
    --region ${AWS_REGION}

# Create repository for Lab 9
aws ecr create-repository \
    --repository-name lab9-tool-calling-api \
    --region ${AWS_REGION}
```

---

## Building and Pushing Docker Images

### Lab 8 Deployment

```bash
cd lab8_deployment

# Build the Docker image
docker build -t lab8-openai-api:latest .

# Tag the image for ECR
docker tag lab8-openai-api:latest ${ECR_REGISTRY}/lab8-openai-api:latest

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Push the image
docker push ${ECR_REGISTRY}/lab8-openai-api:latest
```

### Lab 9 Deployment

```bash
cd ../lab9_deployment

# Build the Docker image
docker build -t lab9-tool-calling-api:latest .

# Tag the image for ECR
docker tag lab9-tool-calling-api:latest ${ECR_REGISTRY}/lab9-tool-calling-api:latest

# Push the image
docker push ${ECR_REGISTRY}/lab9-tool-calling-api:latest
```

---

## ECS Deployment

### Step 1: Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name llm-deployment-cluster \
    --region ${AWS_REGION}
```

### Step 2: Create IAM Roles

#### Task Execution Role

```bash
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### Step 3: Create Task Definitions

#### Lab 8 Task Definition (vLLM)

Create `lab8-task-definition.json`:

```json
{
  "family": "lab8-vllm-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "30720",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "vllm-container",
      "image": "${ECR_REGISTRY}/lab8-openai-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "MODEL",
          "value": "meta-llama/Llama-3.1-8B-Instruct"
        },
        {
          "name": "SERVED_MODEL_NAME",
          "value": "llama-3.1-8b-instruct"
        },
        {
          "name": "TENSOR_PARALLEL_SIZE",
          "value": "1"
        },
        {
          "name": "GPU_MEMORY_UTILIZATION",
          "value": "0.90"
        },
        {
          "name": "MAX_MODEL_LEN",
          "value": "4096"
        },
        {
          "name": "MAX_NUM_BATCHED_TOKENS",
          "value": "8192"
        },
        {
          "name": "MAX_NUM_SEQS",
          "value": "256"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lab8-vllm-api",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "vllm"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "ulimits": [
        {
          "name": "memlock",
          "softLimit": -1,
          "hardLimit": -1
        }
      ]
    }
  ]
}
```

**Key Changes for vLLM:**
- Increased memory to 30GB (vLLM needs more RAM for efficient KV cache)
- Added environment variables for vLLM configuration
- Added `ulimits` for memlock (required for GPU memory pinning)
- Updated log group name

Replace placeholders and register:

```bash
# Create CloudWatch log group
aws logs create-log-group \
    --log-group-name /ecs/lab8-vllm-api \
    --region ${AWS_REGION}

# Register task definition (after replacing placeholders)
envsubst < lab8-task-definition.json | \
    aws ecs register-task-definition \
    --cli-input-json file:///dev/stdin \
    --region ${AWS_REGION}
```

#### Lab 9 Task Definitions (Two-Tier: vLLM + Orchestrator)

**Task 1: vLLM Inference Service (GPU)**

Create `lab9-vllm-task-definition.json`:

```json
{
  "family": "lab9-vllm-inference",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "30720",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "vllm-inference",
      "image": "${ECR_REGISTRY}/lab9-vllm:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "MODEL",
          "value": "meta-llama/Llama-3.1-8B-Instruct"
        },
        {
          "name": "TENSOR_PARALLEL_SIZE",
          "value": "1"
        },
        {
          "name": "GPU_MEMORY_UTILIZATION",
          "value": "0.90"
        },
        {
          "name": "MAX_MODEL_LEN",
          "value": "4096"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lab9-vllm-inference",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "vllm"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "ulimits": [
        {
          "name": "memlock",
          "softLimit": -1,
          "hardLimit": -1
        }
      ]
    }
  ]
}
```

**Task 2: FastAPI Orchestrator (CPU)**

Create `lab9-orchestrator-task-definition.json`:

```json
{
  "family": "lab9-orchestrator",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "orchestrator",
      "image": "${ECR_REGISTRY}/lab9-orchestrator:latest",
      "portMappings": [
        {
          "containerPort": 8001,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "VLLM_BASE_URL",
          "value": "http://lab9-vllm.local:8000/v1"
        },
        {
          "name": "WEATHER_API_KEY",
          "value": "demo-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lab9-orchestrator",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "orchestrator"
        }
      }
    }
  ]
}
```

**Note on Networking**: The orchestrator needs to discover the vLLM service. We'll use AWS Cloud Map for service discovery.

Register both task definitions:

```bash
# Create CloudWatch log groups
aws logs create-log-group \
    --log-group-name /ecs/lab9-vllm-inference \
    --region ${AWS_REGION}

aws logs create-log-group \
    --log-group-name /ecs/lab9-orchestrator \
    --region ${AWS_REGION}

# Register vLLM task definition
envsubst < lab9-vllm-task-definition.json | \
    aws ecs register-task-definition \
    --cli-input-json file:///dev/stdin \
    --region ${AWS_REGION}

# Register orchestrator task definition
envsubst < lab9-orchestrator-task-definition.json | \
    aws ecs register-task-definition \
    --cli-input-json file:///dev/stdin \
    --region ${AWS_REGION}
```

**Set Up Service Discovery (Cloud Map)**

```bash
# Create private DNS namespace
aws servicediscovery create-private-dns-namespace \
    --name local \
    --vpc ${VPC_ID} \
    --region ${AWS_REGION}

# Get namespace ID
NAMESPACE_ID=$(aws servicediscovery list-namespaces \
    --filters Name=TYPE,Values=DNS_PRIVATE \
    --query "Namespaces[?Name=='local'].Id" \
    --output text)

# Create service discovery service for vLLM
aws servicediscovery create-service \
    --name lab9-vllm \
    --dns-config "NamespaceId=${NAMESPACE_ID},DnsRecords=[{Type=A,TTL=10}]" \
    --health-check-custom-config FailureThreshold=1 \
    --region ${AWS_REGION}

# Get service ARN
DISCOVERY_SERVICE_ARN=$(aws servicediscovery list-services \
    --filters Name=NAMESPACE_ID,Values=${NAMESPACE_ID} \
    --query "Services[?Name=='lab9-vllm'].Arn" \
    --output text)
```

### Step 4: Create ECS Services

First, you need to set up networking (VPC, subnets, security groups). Assuming you have a default VPC:

```bash
# Get default VPC ID
export VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" \
    --output text)

# Get subnet IDs
export SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].SubnetId" \
    --output text | tr '\t' ',')

# Create security group
export SG_ID=$(aws ec2 create-security-group \
    --group-name llm-api-sg \
    --description "Security group for LLM API" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' \
    --output text)

# Allow inbound traffic on port 8000
aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0
```

Create services:

```bash
# Lab 8 Service (vLLM)
aws ecs create-service \
    --cluster llm-deployment-cluster \
    --service-name lab8-vllm-service \
    --task-definition lab8-vllm-api \
    --desired-count 1 \
    --launch-type EC2 \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
    --region ${AWS_REGION}

# Lab 9 - Service 1: vLLM Inference (GPU, EC2)
aws ecs create-service \
    --cluster llm-deployment-cluster \
    --service-name lab9-vllm-inference \
    --task-definition lab9-vllm-inference \
    --desired-count 1 \
    --launch-type EC2 \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=DISABLED}" \
    --service-registries "registryArn=${DISCOVERY_SERVICE_ARN}" \
    --region ${AWS_REGION}

# Lab 9 - Service 2: Orchestrator (CPU, Fargate)
aws ecs create-service \
    --cluster llm-deployment-cluster \
    --service-name lab9-orchestrator \
    --task-definition lab9-orchestrator \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
    --region ${AWS_REGION}
```

**Key Points:**
- **vLLM service**: Runs on GPU EC2 instances, registered with Cloud Map for service discovery
- **Orchestrator service**: Runs on Fargate (cheaper CPU), can scale independently (2+ instances)
- Orchestrator discovers vLLM via DNS: `lab9-vllm.local:8000`

**Note:** You'll need to launch GPU-enabled EC2 instances in your ECS cluster. Consider using an ECS-optimized AMI with GPU support.

---

## Testing the Deployment

### Find Your Service Endpoints

```bash
# Get task ARN for Lab 8
TASK_ARN=$(aws ecs list-tasks \
    --cluster llm-deployment-cluster \
    --service-name lab8-vllm-service \
    --query 'taskArns[0]' \
    --output text)

# Get task details and public IP
ENI_ID=$(aws ecs describe-tasks \
    --cluster llm-deployment-cluster \
    --tasks ${TASK_ARN} \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

# Get public IP from ENI
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids ${ENI_ID} \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

echo "vLLM Service URL: http://${PUBLIC_IP}:8000"
```

### Test Lab 8 (vLLM Basic API)

```python
from openai import OpenAI
import time

# Initialize client
client = OpenAI(
    base_url="http://<PUBLIC_IP>:8000/v1",
    api_key="not-needed"  # vLLM doesn't require API key by default
)

# Test 1: Basic chat completion
print("=" * 60)
print("Test 1: Basic Chat Completion")
print("=" * 60)

start = time.time()
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Explain quantization in machine learning in 2-3 sentences."}
    ],
    max_tokens=150,
    temperature=0.7
)
elapsed = time.time() - start

print(f"Response: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Latency: {elapsed:.2f}s")
print(f"Throughput: {response.usage.completion_tokens / elapsed:.2f} tokens/s")

# Test 2: Streaming
print("\n" + "=" * 60)
print("Test 2: Streaming Response")
print("=" * 60)

start = time.time()
stream = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Count from 1 to 10."}
    ],
    max_tokens=100,
    stream=True
)

print("Streaming: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print(f"\nStreaming latency: {time.time() - start:.2f}s")

# Test 3: Check available models
print("\n" + "=" * 60)
print("Test 3: List Available Models")
print("=" * 60)
models = client.models.list()
for model in models.data:
    print(f"Model: {model.id}")
```

### Test Lab 9 (Two-Tier Tool Calling)

The orchestrator handles all tool calling logic - you just send requests like normal OpenAI API calls.

```python
import requests
import json
import time

# Get orchestrator public IP
# For Lab 9, you connect to the ORCHESTRATOR (port 8001), not vLLM directly
ORCHESTRATOR_URL = "http://<ORCHESTRATOR_PUBLIC_IP>:8001"

# Test 1: Multi-turn tool calling (weather + math)
print("=" * 60)
print("Test 1: Multi-Tool Query")
print("=" * 60)

start = time.time()
response = requests.post(
    f"{ORCHESTRATOR_URL}/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco and what is 15 times 23?"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
)
elapsed = time.time() - start

result = response.json()
print(f"Response: {result['choices'][0]['message']['content']}")
print(f"Tokens: {result['usage']['total_tokens']}")
print(f"Tool Rounds: {result['metadata']['tool_rounds']}")
print(f"Total Latency: {elapsed:.2f}s")

# Test 2: List available tools
print("\n" + "=" * 60)
print("Test 2: Available Tools")
print("=" * 60)

response = requests.get(f"{ORCHESTRATOR_URL}/tools")
tools = response.json()
for tool in tools['tools']:
    print(f"- {tool['function']['name']}: {tool['function']['description']}")

# Test 3: Database search tool
print("\n" + "=" * 60)
print("Test 3: Database Search")
print("=" * 60)

response = requests.post(
    f"{ORCHESTRATOR_URL}/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": "Search for products in the database"
            }
        ],
        "max_tokens": 300
    }
)

result = response.json()
print(f"Response: {result['choices'][0]['message']['content']}")

# Test 4: Concurrent requests (test orchestrator scaling)
print("\n" + "=" * 60)
print("Test 4: Concurrent Requests")
print("=" * 60)

import concurrent.futures

def make_request(i):
    start = time.time()
    response = requests.post(
        f"{ORCHESTRATOR_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": f"What is {i} times 10?"}],
            "max_tokens": 100
        }
    )
    return time.time() - start

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    latencies = list(executor.map(make_request, range(10)))

print(f"10 concurrent requests completed in {time.time() - start:.2f}s")
print(f"Average latency: {sum(latencies) / len(latencies):.2f}s")
print(f"Requests/sec: {10 / (time.time() - start):.2f}")
```

**Architecture Benefits Demonstrated:**
- Orchestrator handles tool execution transparently
- Can scale orchestrator independently (2+ instances for load balancing)
- GPU (vLLM) only used for inference, not waiting for tool execution
- Easy to add new tools without touching inference layer

### Benchmarking vLLM Performance

```python
import asyncio
import time
from openai import OpenAI

client = OpenAI(base_url="http://<PUBLIC_IP>:8000/v1", api_key="not-needed")

async def benchmark_concurrent_requests(num_requests=10):
    """Test vLLM's continuous batching capabilities"""

    async def single_request(i):
        start = time.time()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": f"Tell me a short fact about number {i}."}],
            max_tokens=50
        )
        return time.time() - start

    start = time.time()
    tasks = [single_request(i) for i in range(num_requests)]
    latencies = await asyncio.gather(*tasks)
    total_time = time.time() - start

    print(f"Concurrent Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Latency: {sum(latencies)/len(latencies):.2f}s")
    print(f"Requests/sec: {num_requests/total_time:.2f}")

# Run benchmark
asyncio.run(benchmark_concurrent_requests(10))
```

---

## Monitoring and Scaling

### CloudWatch Logs

View logs in CloudWatch:

```bash
# Lab 8 logs
aws logs tail /ecs/lab8-vllm-api --follow

# Lab 9 logs
aws logs tail /ecs/lab9-vllm-tool-calling-api --follow
```

### vLLM Metrics Endpoint

vLLM exposes Prometheus-compatible metrics at `/metrics`:

```bash
curl http://<PUBLIC_IP>:8000/metrics
```

**Key Metrics to Monitor:**

```
# Request metrics
vllm:num_requests_running          # Currently processing requests
vllm:num_requests_waiting          # Queued requests
vllm:avg_prompt_throughput_toks_per_s    # Prompt processing speed
vllm:avg_generation_throughput_toks_per_s # Generation speed

# GPU metrics
vllm:gpu_cache_usage_perc         # KV cache utilization
vllm:gpu_prefix_cache_hit_rate    # Prefix cache efficiency

# Performance metrics
vllm:time_to_first_token_seconds  # Latency to first token
vllm:time_per_output_token_seconds # Per-token generation latency
vllm:e2e_request_latency_seconds  # End-to-end latency
```

### Setting Up CloudWatch Dashboard

Create a custom dashboard for vLLM metrics:

```bash
cat > vllm-dashboard.json <<EOF
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", {"stat": "Average"}],
          [".", "MemoryUtilization", {"stat": "Average"}]
        ],
        "period": 60,
        "stat": "Average",
        "region": "${AWS_REGION}",
        "title": "ECS Resource Utilization"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
    --dashboard-name vllm-deployment \
    --dashboard-body file://vllm-dashboard.json
```

### Auto Scaling

For vLLM deployments, scale based on **request queue depth** rather than CPU:

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/llm-deployment-cluster/lab8-vllm-service \
    --min-capacity 1 \
    --max-capacity 5

# Create custom metric for queue depth (requires Prometheus/CloudWatch integration)
# For CPU-based scaling (simple approach):
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/llm-deployment-cluster/lab8-vllm-service \
    --policy-name gpu-memory-scaling-policy \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

**scaling-policy.json** (GPU memory-based):
```json
{
  "TargetValue": 85.0,
  "CustomizedMetricSpecification": {
    "MetricName": "MemoryUtilization",
    "Namespace": "AWS/ECS",
    "Statistic": "Average"
  },
  "ScaleInCooldown": 300,
  "ScaleOutCooldown": 60
}
```

**Best Practices for vLLM Autoscaling:**
1. Scale based on queued requests (`vllm:num_requests_waiting`)
2. Set scale-out cooldown low (60s) for burst traffic
3. Set scale-in cooldown high (300s) to avoid flapping
4. Consider using target tracking on custom CloudWatch metrics from Prometheus

---

## Cleanup

To avoid ongoing charges, delete all resources:

```bash
# Delete services
aws ecs update-service \
    --cluster llm-deployment-cluster \
    --service lab8-vllm-service \
    --desired-count 0

aws ecs delete-service \
    --cluster llm-deployment-cluster \
    --service lab8-vllm-service

aws ecs update-service \
    --cluster llm-deployment-cluster \
    --service lab9-vllm-tool-service \
    --desired-count 0

aws ecs delete-service \
    --cluster llm-deployment-cluster \
    --service lab9-vllm-tool-service

# Wait for services to be deleted
aws ecs wait services-inactive \
    --cluster llm-deployment-cluster \
    --services lab8-vllm-service lab9-vllm-tool-service

# Delete cluster
aws ecs delete-cluster \
    --cluster llm-deployment-cluster

# Delete ECR repositories
aws ecr delete-repository \
    --repository-name lab8-openai-api \
    --force

aws ecr delete-repository \
    --repository-name lab9-tool-calling-api \
    --force

# Delete log groups
aws logs delete-log-group --log-group-name /ecs/lab8-vllm-api
aws logs delete-log-group --log-group-name /ecs/lab9-vllm-tool-calling-api

# Delete security group
aws ec2 delete-security-group --group-id ${SG_ID}

# Deregister task definitions (optional - they don't incur charges)
# List all revisions
aws ecs list-task-definitions --family-prefix lab8-vllm-api
aws ecs list-task-definitions --family-prefix lab9-vllm-tool-calling-api
```

---

## Troubleshooting

### Common vLLM Issues

#### 1. Container Fails to Start

Check logs for initialization errors:
```bash
aws logs tail /ecs/lab8-vllm-api --follow --since 10m
```

**Common error**: "CUDA out of memory"
- **Solution**: Reduce `gpu_memory_utilization` from 0.90 to 0.85
- **Solution**: Decrease `max_model_len` or `max_num_batched_tokens`

#### 2. Model Loading Takes Too Long (>3 minutes)

**Causes:**
- Model downloading from HuggingFace Hub
- Large model size

**Solutions:**
```dockerfile
# Pre-download model in Dockerfile
FROM vllm/vllm-openai:latest

# Cache model weights
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('meta-llama/Llama-3.1-8B-Instruct', \
    cache_dir='/root/.cache/huggingface')"

ENV HF_HOME=/root/.cache/huggingface
```

Or use EFS/EBS volumes for model caching across tasks.

#### 3. Low Throughput / High Latency

Check vLLM metrics:
```bash
curl http://<PUBLIC_IP>:8000/metrics | grep throughput
```

**Optimizations:**
```bash
# Increase concurrent sequences
--max-num-seqs 512

# Enable chunked prefill
--enable-chunked-prefill

# Adjust batch size
--max-num-batched-tokens 16384

# For multi-GPU (g4dn.12xlarge)
--tensor-parallel-size 4
```

#### 4. GPU Not Detected

Verify GPU is available:
```bash
# SSH into EC2 instance
nvidia-smi

# Check ECS agent can see GPU
cat /var/lib/ecs/gpu/nvidia_gpu_info.json
```

**Fix**: Ensure using ECS GPU-optimized AMI:
```bash
# Get latest ECS GPU AMI
aws ssm get-parameter \
    --name /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended \
    --region ${AWS_REGION}
```

#### 5. OOM (Out of Memory) During Inference

**Symptom**: Requests fail mid-generation

**Solutions:**
1. Reduce `max_model_len`: `--max-model-len 2048`
2. Reduce `gpu_memory_utilization`: `--gpu-memory-utilization 0.85`
3. Decrease `max_num_seqs`: `--max-num-seqs 128`
4. Enable memory profiling: `--enable-prefix-caching`

#### 6. Tool Calling Not Working (Lab 9)

**Check model compatibility:**
```bash
# Only these models support tool calling well:
# - meta-llama/Llama-3.1-*-Instruct
# - mistralai/Mistral-*-Instruct-*
# - Qwen/Qwen2.5-*-Instruct
```

**Verify parser:**
```bash
# Try different parsers
--tool-call-parser llama3_json  # For Llama 3.1
--tool-call-parser mistral      # For Mistral
--tool-call-parser hermes       # For general use
```

---

## Performance Optimization

### 1. Tuning vLLM Parameters

#### For Low Latency (Interactive Applications)
```bash
--max-num-seqs 64 \
--gpu-memory-utilization 0.85 \
--max-model-len 2048 \
--enable-chunked-prefill
```

#### For High Throughput (Batch Processing)
```bash
--max-num-seqs 512 \
--gpu-memory-utilization 0.95 \
--max-num-batched-tokens 32768 \
--enable-chunked-prefill
```

#### For Long Context
```bash
--max-model-len 32768 \
--gpu-memory-utilization 0.90 \
--enable-chunked-prefill \
--max-num-batched-tokens 8192 \
--max-num-seqs 32
```

### 2. Multi-GPU Configuration

For `g4dn.12xlarge` (4x T4 GPUs):
```bash
--tensor-parallel-size 4 \
--max-num-seqs 1024 \
--max-num-batched-tokens 65536
```

**Expected Performance Gains:**
- 3.5-4x throughput increase
- Same latency as single GPU
- Can serve larger models (up to 30B parameters)

### 3. Quantization for Better Performance

Use AWQ or GPTQ quantized models:
```bash
# AWQ (better quality)
--model TheBloke/Llama-2-13B-AWQ \
--quantization awq

# GPTQ (better compatibility)
--model TheBloke/Llama-2-13B-GPTQ \
--quantization gptq
```

**Benefits:**
- 2-3x higher throughput
- 50% less VRAM usage
- Can fit larger models on same GPU

### 4. Prefix Caching

Enable for repetitive prompts (e.g., system prompts):
```bash
--enable-prefix-caching
```

**Use Case:** When system prompts are identical across requests
**Performance Gain:** 20-40% faster for repeated prefixes

### 5. Speculative Decoding (vLLM 0.5.0+)

For faster generation with smaller draft model:
```bash
--speculative-model meta-llama/Llama-3.1-1B-Instruct \
--num-speculative-tokens 5
```

**Performance Gain:** 1.5-2x faster generation

---

## Production Best Practices

### 1. Model Warming

Add model warming to prevent cold start latency:
```python
# Add to entrypoint script
import requests
import time

time.sleep(30)  # Wait for vLLM to start

# Warm up model
requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5
})
```

### 2. Rate Limiting

Add nginx reverse proxy for rate limiting:
```nginx
limit_req_zone $binary_remote_addr zone=llm:10m rate=10r/s;

server {
    listen 80;
    location / {
        limit_req zone=llm burst=20 nodelay;
        proxy_pass http://localhost:8000;
    }
}
```

### 3. Request Timeout Configuration

```bash
# Client-side timeout
--timeout-keep-alive 300

# Server-side max generation time
--max-total-tokens 4096
```

### 4. Monitoring Alerts

Set up CloudWatch alarms:
```bash
# High queue depth (scale out trigger)
aws cloudwatch put-metric-alarm \
    --alarm-name vllm-high-queue \
    --metric-name num_requests_waiting \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold

# High GPU memory
aws cloudwatch put-metric-alarm \
    --alarm-name vllm-high-gpu-mem \
    --metric-name gpu_cache_usage_perc \
    --threshold 95 \
    --comparison-operator GreaterThanThreshold
```

### 5. Cost Optimization

#### Use Spot Instances
Save 70% on compute costs:
```bash
# Update task definition to use Spot capacity provider
aws ecs create-capacity-provider \
    --name gpu-spot-provider \
    --auto-scaling-group-provider "autoScalingGroupArn=arn:aws:autoscaling:...,
        managedScaling={status=ENABLED,targetCapacity=100},
        managedTerminationProtection=DISABLED" \
    --capacity-provider-strategy capacityProvider=gpu-spot-provider,weight=1,base=0
```

#### Right-Size Instances
- **7B models**: g4dn.xlarge (1x T4) - $0.526/hr
- **13B models**: g4dn.2xlarge (1x T4) - $0.752/hr or g5.xlarge (1x A10G) - $1.006/hr
- **30B+ models**: g4dn.12xlarge (4x T4) - $3.912/hr or g5.12xlarge (4x A10G) - $5.672/hr

#### Model Selection
- Use smaller models when possible (Llama 3.1 8B vs 70B)
- Quantize to AWQ/GPTQ for 50% VRAM reduction
- Consider distilled models for simple tasks

### 6. Security Hardening

```bash
# Add authentication middleware
--api-key your-secret-api-key

# Enable HTTPS with ALB
# Use AWS Secrets Manager for API keys
# Implement request validation
# Add AWS WAF for DDoS protection
```

---

## Additional Resources

### vLLM Documentation
- [vLLM Official Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [vLLM Performance Benchmarks](https://blog.vllm.ai/2023/06/20/vllm.html)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### AWS Documentation
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [ECS GPU Support](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)
- [ECS Task Definitions](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html)
- [ECR Documentation](https://docs.aws.amazon.com/ecr/)

### LLM Resources
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/)

---

## Appendix: Alternative Deployment Options

### Using Application Load Balancer (Production)

For production deployments with HTTPS and multiple instances:

```bash
# 1. Create target group
aws elbv2 create-target-group \
    --name vllm-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id ${VPC_ID} \
    --target-type ip \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3

# 2. Create ALB
aws elbv2 create-load-balancer \
    --name vllm-alb \
    --subnets ${SUBNET_IDS} \
    --security-groups ${SG_ID} \
    --scheme internet-facing \
    --type application

# 3. Create HTTPS listener (requires ACM certificate)
aws elbv2 create-listener \
    --load-balancer-arn ${ALB_ARN} \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=${CERT_ARN} \
    --default-actions Type=forward,TargetGroupArn=${TG_ARN}

# 4. Update ECS service to use ALB
aws ecs update-service \
    --cluster llm-deployment-cluster \
    --service lab8-vllm-service \
    --load-balancers targetGroupArn=${TG_ARN},containerName=vllm-container,containerPort=8000
```

### Multi-Region Deployment

For global low-latency access:

1. Deploy to multiple AWS regions (us-east-1, eu-west-1, ap-southeast-1)
2. Use Route53 latency-based routing
3. Replicate ECR images across regions
4. Implement regional failover

```bash
# Route53 latency-based routing
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch file://route53-latency.json
```

### Kubernetes/EKS Deployment

For Kubernetes users, vLLM can be deployed on EKS:

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-3.1-8B-Instruct
          - --tensor-parallel-size
          - "1"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 30Gi
          requests:
            nvidia.com/gpu: 1
            memory: 30Gi
```

### Using AWS SageMaker

Alternative to ECS using SageMaker for managed deployment:

```python
from sagemaker.huggingface import HuggingFaceModel

# Define model
huggingface_model = HuggingFaceModel(
    model_data="s3://my-bucket/model.tar.gz",
    role=role,
    image_uri="vllm/vllm-openai:latest",
    env={
        'MODEL_NAME': 'meta-llama/Llama-3.1-8B-Instruct',
        'TENSOR_PARALLEL_SIZE': '1'
    }
)

# Deploy
predictor = huggingface_model.deploy(
    instance_type="ml.g4dn.xlarge",
    initial_instance_count=1
)
```

### CI/CD Integration

Set up automated deployments with GitHub Actions:

```yaml
# .github/workflows/deploy.yml
name: Deploy vLLM to ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REGISTRY}

      - name: Build and push image
        run: |
          docker build -t lab8-openai-api:${{ github.sha }} ./lab8_deployment
          docker tag lab8-openai-api:${{ github.sha }} ${ECR_REGISTRY}/lab8-openai-api:latest
          docker push ${ECR_REGISTRY}/lab8-openai-api:latest

      - name: Update ECS service
        run: |
          aws ecs update-service --cluster llm-deployment-cluster --service lab8-vllm-service --force-new-deployment
```

---

## Performance Expectations

### Expected Throughput (Llama 3.1 8B on g4dn.xlarge)

| Configuration | Throughput (tokens/sec) | Concurrent Users |
|--------------|-------------------------|------------------|
| Low Latency  | ~50-80                 | 1-10             |
| Balanced     | ~120-150               | 10-50            |
| High Throughput | ~200-300            | 50-200           |

### Expected Latency

| Metric | Low Latency Mode | High Throughput Mode |
|--------|------------------|---------------------|
| Time to First Token | 30-50ms | 100-200ms |
| Per Token Latency | 10-15ms | 5-8ms |
| Total (100 tokens) | 1-2s | 0.8-1.2s |

### Cost Estimates (us-east-1)

| Instance Type | Cost/Hour | Requests/Hour (est.) | Cost/1M Requests |
|--------------|-----------|----------------------|------------------|
| g4dn.xlarge  | $0.526    | 3,600-7,200         | $73-146          |
| g4dn.2xlarge | $0.752    | 5,400-10,800        | $70-139          |
| g5.xlarge    | $1.006    | 4,800-9,600         | $105-210         |

*Assumes average 100 tokens/request, 30s processing time*

---

---

## Enterprise-Scale Deployment Patterns

### Ray Serve for High-Scale Production (>1k QPS)

For enterprise deployments serving thousands of requests per second, Ray Serve provides advanced orchestration:

#### Why Ray Serve at Scale?

- **Distributed orchestration**: Route across multiple vLLM instances
- **Multi-model serving**: Serve different models/adapters simultaneously
- **A/B testing**: Gradual rollout of new models
- **Advanced autoscaling**: Custom metrics-based scaling
- **Request routing**: Tenant-aware routing, priority queues
- **Fault tolerance**: Automatic failover and retries

#### Ray Serve Configuration

```python
# ray_serve_config.py
from ray import serve
from ray.serve.handle import DeploymentHandle
import httpx

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_cpus": 2}
)
class OrchestrationService:
    def __init__(self, vllm_handle: DeploymentHandle):
        self.vllm = vllm_handle
        self.tools = load_tools()  # Tenant-specific tools

    async def __call__(self, request):
        # Multi-tenant routing logic
        tenant_id = request.headers.get("X-Tenant-ID")
        tools = self.tools.get(tenant_id, [])

        # Call vLLM for inference
        result = await self.vllm.remote(request)

        # Execute tools based on tenant config
        if result.needs_tool:
            tool_result = await self.execute_tool(
                result.tool_name,
                result.tool_args,
                tenant_id
            )
            # Get final response
            final = await self.vllm.remote(
                request.with_tool_result(tool_result)
            )
            return final

        return result

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1}
)
class vLLMDeployment:
    def __init__(self):
        from vllm import LLM
        self.model = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=1
        )

    def __call__(self, request):
        return self.model.generate(request)

# Deploy
vllm = vLLMDeployment.bind()
orchestrator = OrchestrationService.bind(vllm)
serve.run(orchestrator)
```

#### Ray Serve on ECS

Deploy Ray cluster on ECS with:
- **Head node**: Manages cluster state
- **Worker nodes**: Run vLLM and orchestration replicas
- **Service discovery**: Use AWS Cloud Map
- **Autoscaling**: CloudWatch metrics → ECS autoscaling

---

## Multi-Tenant Deployment Patterns

### Pattern 1: Shared Inference, Isolated Tools

```python
# tenant_config.py
TENANT_CONFIGS = {
    "org_a": {
        "tools": ["crm_search", "email_send"],
        "rate_limit": 1000,  # requests/min
        "model": "llama-3.1-8b"
    },
    "org_b": {
        "tools": ["database_query", "analytics"],
        "rate_limit": 5000,
        "model": "llama-3.1-8b"
    }
}

# In orchestrator
async def execute_tool(self, tool_name, args, tenant_id):
    config = TENANT_CONFIGS[tenant_id]

    # Security: only allow tenant's tools
    if tool_name not in config["tools"]:
        raise PermissionError(f"Tool {tool_name} not available")

    # Execute with tenant isolation
    return await TOOLS[tool_name](args, tenant_context=tenant_id)
```

### Pattern 2: LoRA Multi-Tenant (Different Models per Tenant)

vLLM supports serving multiple LoRA adapters efficiently:

```bash
# Start vLLM with LoRA support
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules org_a=/models/org_a_lora \
                    org_b=/models/org_b_lora \
    --max-loras 10
```

Request with specific adapter:
```python
response = client.chat.completions.create(
    model="org_a",  # Routes to org_a LoRA adapter
    messages=[...]
)
```

**Benefits**:
- Share base model weights (save GPU memory)
- Different fine-tuning per tenant
- 10-100x more memory efficient than separate models

---

## Cost Optimization at Scale

### 1. Multi-Region Deployment with Regional Routing

```
Route53 Latency-Based Routing
    ├─→ us-east-1 (ECS cluster)
    ├─→ eu-west-1 (ECS cluster)
    └─→ ap-southeast-1 (ECS cluster)
```

**Benefits**:
- Lower latency (serve from nearest region)
- Cost savings (data transfer within region)
- Compliance (data residency requirements)

### 2. Tiered Inference Strategy

```python
# Cost-aware routing
def route_to_model(request):
    complexity = estimate_complexity(request)

    if complexity < 0.3:
        return "distilled-1b"     # $0.10/hr, faster
    elif complexity < 0.7:
        return "standard-8b"      # $0.50/hr
    else:
        return "large-70b"        # $3.00/hr, slower
```

**Savings**: 60-80% cost reduction for simple queries

### 3. Prefix Caching Strategy

```python
# System prompts shared across org
SYSTEM_PROMPTS = {
    "org_a": "You are a CRM assistant for...",  # 500 tokens
    "org_b": "You are a data analyst for..."    # 600 tokens
}

# vLLM caches these prefixes automatically
# Saves 30-50% on prompt processing costs
```

### 4. Spot Instance Strategy

```bash
# Use Spot for 70% savings on non-critical traffic
aws ecs create-capacity-provider \
    --name gpu-spot \
    --auto-scaling-group-provider \
        "autoScalingGroupArn=arn:...,\
         managedScaling={status=ENABLED,targetCapacity=80},\
         managedTerminationProtection=ENABLED"

# Mix: 80% Spot, 20% On-Demand for reliability
--capacity-provider-strategy \
    capacityProvider=gpu-spot,weight=80,base=0 \
    capacityProvider=gpu-ondemand,weight=20,base=1
```

**Expected Savings**: 55-60% cost reduction

---

## Advanced Monitoring & Observability

### Distributed Tracing

```python
# OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

@app.post("/v1/chat/completions")
async def chat(request):
    with trace.get_tracer(__name__).start_as_current_span("orchestration"):
        # This span tracks end-to-end latency

        with trace.get_tracer(__name__).start_as_current_span("vllm_inference"):
            result = await call_vllm(request)

        if result.needs_tool:
            with trace.get_tracer(__name__).start_as_current_span("tool_execution"):
                tool_result = await execute_tool(result)

        return final_response
```

**Export to**: Jaeger, Zipkin, AWS X-Ray for visualization

### Custom Metrics for Enterprise Scale

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
tool_calls_total = Counter('tool_calls_total', 'Total tool calls', ['tool_name', 'tenant_id'])
tool_latency = Histogram('tool_latency_seconds', 'Tool execution time', ['tool_name'])
active_tenants = Gauge('active_tenants', 'Number of active tenants')

# Track in orchestrator
@app.post("/v1/chat/completions")
async def chat(request):
    tenant_id = request.headers.get("X-Tenant-ID")

    if tool_called:
        tool_calls_total.labels(tool_name=tool_name, tenant_id=tenant_id).inc()
        with tool_latency.labels(tool_name=tool_name).time():
            result = await execute_tool(tool_name)
```

### CloudWatch Dashboard for Operations

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization"],
          ["vLLM", "num_requests_waiting"],
          ["vLLM", "gpu_cache_usage_perc"],
          ["Custom", "tool_calls_per_tenant"]
        ],
        "period": 60,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Enterprise LLM Platform - Operations View"
      }
    }
  ]
}
```

---

## Security & Compliance for Enterprise

### 1. API Key Management with AWS Secrets Manager

```python
import boto3

secrets_client = boto3.client('secretsmanager')

@app.middleware("http")
async def validate_tenant(request, call_next):
    api_key = request.headers.get("X-API-Key")

    # Retrieve from Secrets Manager
    tenant_secret = secrets_client.get_secret_value(
        SecretId=f"tenant-{tenant_id}"
    )

    if api_key != tenant_secret['api_key']:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    return await call_next(request)
```

### 2. Request/Response Logging for Compliance

```python
@app.middleware("http")
async def audit_log(request, call_next):
    request_id = str(uuid.uuid4())

    # Log request
    await log_to_s3({
        "request_id": request_id,
        "tenant_id": request.headers.get("X-Tenant-ID"),
        "timestamp": datetime.utcnow(),
        "endpoint": request.url.path,
        "prompt": request.body  # Encrypted
    })

    response = await call_next(request)

    # Log response
    await log_to_s3({
        "request_id": request_id,
        "response": response.body,  # Encrypted
        "status": response.status_code
    })

    return response
```

### 3. Tenant Data Isolation

```python
# Ensure tools can't access cross-tenant data
class TenantIsolationError(Exception):
    pass

async def execute_tool(tool_name, args, tenant_id):
    # Add tenant_id to all DB queries
    if tool_name == "database_query":
        # CRITICAL: Prevent SQL injection and tenant data leakage
        args['where_clause'] += f" AND tenant_id = '{tenant_id}'"

    # Validate no cross-tenant access
    result = await TOOLS[tool_name](args)

    if not validate_tenant_ownership(result, tenant_id):
        raise TenantIsolationError("Cross-tenant access detected")

    return result
```

---

## Conclusion

This deployment guide provides production and enterprise-ready patterns for serving LLMs using vLLM on AWS ECS.

### Key Takeaways by Scale

**For Learning/Small Scale (<100 QPS)**:
1. Single vLLM instance with FastAPI orchestrator
2. Single region, single model
3. Manual scaling

**For Production (<1k QPS)**:
1. Multi-instance vLLM behind ALB
2. FastAPI with Redis caching
3. Auto-scaling based on queue depth
4. Multi-region for redundancy

**For Enterprise Scale (>1k QPS)**:
1. Ray Serve for distributed orchestration
2. LoRA multi-tenancy for memory efficiency
3. Multi-region with latency-based routing
4. Tiered inference (distilled vs full models)
5. Comprehensive monitoring (OpenTelemetry + Prometheus)
6. 55-60% cost savings with Spot instances

### Technical Decisions

- **vLLM**: Industry standard for inference (24x throughput vs naive)
- **Two-tier architecture**: Separates business logic from inference
- **GPU instances**: See hardware requirements table below
- **Quantization**: AWQ/GPTQ for 50% memory reduction
- **Monitoring**: vLLM metrics + custom business metrics

### Hardware Requirements for Large Models

| Model Size | Precision | VRAM Needed | AWS Instance | GPUs | Cost/Hour | Throughput |
|------------|-----------|-------------|--------------|------|-----------|------------|
| **7-8B** | FP16 | ~16GB | g5.xlarge | 1x A10G (24GB) | $1.01 | ~50-80 tok/s |
| **7-8B** | AWQ-4bit | ~8GB | g4dn.xlarge | 1x T4 (16GB) | $0.53 | ~40-60 tok/s |
| **13B** | FP16 | ~26GB | g5.2xlarge | 1x A10G (24GB) | $1.21 | ~30-50 tok/s |
| **13B** | AWQ-4bit | ~13GB | g4dn.xlarge | 1x T4 (16GB) | $0.53 | ~25-40 tok/s |
| **70B** | FP16 | ~140GB | p4d.24xlarge | 2x A100 (80GB) | $32.77 | ~15-25 tok/s |
| **70B** | AWQ-4bit | ~35GB | g5.12xlarge | 2x A10G (48GB total) | $5.67 | ~10-20 tok/s |
| **238B** | FP16 | ~476GB | p5.48xlarge | 8x H100 (640GB total) | $98.32 | ~8-12 tok/s |
| **238B** | AWQ-4bit | ~119GB | p4d.24xlarge | 2x A100 (160GB total) | $32.77 | ~5-8 tok/s |

**Notes**:
- VRAM includes model weights + KV cache overhead
- Throughput assumes optimal vLLM configuration
- Multi-GPU uses tensor parallelism (splits model across GPUs)
- Prices are us-east-1 On-Demand (Spot can save 70%)

### Ultra-Large Model Deployment (70B+)

For models like **Llama 3.1 70B** or **405B** at enterprise scale:

#### Option 1: Multi-GPU with Tensor Parallelism

```bash
# Llama 70B on 2x A100 80GB (p4d.24xlarge)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --dtype float16
```

**Performance**:
- First token: ~200-300ms
- Tokens/sec: 15-25 (high throughput mode)
- Concurrent requests: 50-100

#### Option 2: Quantization for Cost Savings

```bash
# Llama 70B AWQ-4bit on 2x A10G (g5.12xlarge)
vllm serve TheBloke/Llama-2-70B-AWQ \
    --quantization awq \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
```

**Cost savings**: 83% cheaper ($5.67/hr vs $32.77/hr)
**Performance hit**: ~20-30% slower, but still production-viable

#### Option 3: Pipeline Parallelism (for 238B+)

```bash
# 238B model on 8x H100 (p5.48xlarge)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --dtype bfloat16
```

**When to use 238B+ models**:
- Complex reasoning tasks
- Multi-step problem solving
- Code generation at scale
- Research applications

**Reality check**: Most production use cases (>90%) are well-served by 7-13B models with good fine-tuning. 70B+ is for specialized needs.

### Cost-Effective Strategy for Multiple Model Sizes

```python
# Route by complexity
routing_logic = {
    "simple": "llama-3.1-8b",      # $1.01/hr, 80% of requests
    "moderate": "llama-3.1-70b",   # $32.77/hr, 15% of requests
    "complex": "llama-3.1-405b"    # $98.32/hr, 5% of requests
}

# Estimate savings
baseline_cost = 100 * $98.32  # All requests on 405B
optimized_cost = (80 * $1.01) + (15 * $32.77) + (5 * $98.32)
# Savings: ~95% cost reduction!
```

### AWS Instance Type Selection Guide

**For 7-13B models (most common)**:
- **Development**: g4dn.xlarge (T4, $0.53/hr)
- **Production**: g5.xlarge/2xlarge (A10G, $1.01-1.21/hr)
- **High throughput**: g5.12xlarge (4x A10G, $5.67/hr)

**For 70B models**:
- **FP16**: p4d.24xlarge (8x A100, use 2 with TP=2, $32.77/hr)
- **AWQ-4bit**: g5.12xlarge (4x A10G, use 2 with TP=2, $5.67/hr)

**For 238B+ models**:
- **FP16**: p5.48xlarge (8x H100, $98.32/hr)
- **AWQ-4bit**: p4d.24xlarge (8x A100, use 2-4 with TP, $32.77/hr)

**Spot instance strategy**:
```bash
# Mix for reliability: 80% Spot + 20% On-Demand
# Expected savings: 55-60% total cost reduction
# Use On-Demand for latency-sensitive traffic
# Use Spot for batch processing, non-critical workloads
```

### Cost Benchmarks (at scale)

| Component | Cost/Hour | Optimization | Savings |
|-----------|-----------|-------------|---------|
| g4dn.xlarge (On-Demand) | $0.526 | Spot instances | 70% |
| Orchestrator (c5.xlarge) | $0.17 | Fargate Spot | 60% |
| Data transfer | $0.09/GB | Regional routing | 40% |
| LoRA adapters | +0% | vs separate models | 90% |

**Total potential savings**: 55-65% with proper optimization

For questions, refer to [vLLM Discord](https://discord.gg/vllm) or [GitHub Issues](https://github.com/vllm-project/vllm/issues).
