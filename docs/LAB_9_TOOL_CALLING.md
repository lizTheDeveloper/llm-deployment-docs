# Lab 9: FastAPI Tool Calling with vLLM on AWS

Build a production two-tier architecture: FastAPI orchestration layer + vLLM inference engine for LLM tool calling, deployed to AWS EC2.

**Time:** 45 minutes
**Prerequisites:** Completed [Lab 8](LAB_8_VLLM_DEPLOYMENT.md), AWS EC2 GPU instance with Docker
**What You'll Build:** Multi-service deployment with tool calling (weather, calculator) and vLLM backend running on AWS

---

## What You'll Learn

- Design two-tier LLM architectures (orchestration + inference)
- Implement tool calling with FastAPI
- Deploy multi-service applications with Docker Compose on AWS
- Separate business logic from inference for independent scaling
- Deploy to AWS ECS for production use

---

## Architecture

```
Client (OpenAI SDK/curl)
    ↓
FastAPI Orchestrator (Port 8001)
    ├── Tool Execution (weather, calculator)
    ├── Request Handling
    └── Response Formatting
    ↓
vLLM Inference Engine (Port 8000)
    └── GPU Inference (vLLM v1 engine)
```

**Why Two Tiers?**

| Concern | FastAPI Layer (CPU) | vLLM Layer (GPU) |
|---------|-------------------|------------------|
| **Role** | Business logic, tool calling, RAG | Model inference |
| **Scaling** | Horizontal (replicas) | Vertical (GPU count) |
| **Updates** | Frequent (tools, logic) | Rare (model changes) |
| **Cost** | $0.05/hour (CPU) | $1.01/hour (GPU) |

This separation is how production systems at Anthropic, OpenAI, and Google structure tool-calling capabilities.

---

## Step 1: Understand the Tool Orchestrator

The FastAPI orchestrator (`tool_orchestrator.py`) handles:

1. **Receives requests** with tool definitions
2. **Calls vLLM** for initial inference
3. **Detects tool calls** in the response
4. **Executes tools** (weather API, calculator)
5. **Calls vLLM again** with tool results
6. **Returns final response** to client

### Review the Code

Read the provided `tool_orchestrator.py` (already created in docs/):

```python
# Key components:
# 1. Tool definitions (get_current_weather, calculate_math)
# 2. OpenAI-compatible API endpoint
# 3. Tool call detection logic
# 4. Two-pass inference (initial + tool result)
```

---

## Step 2: Create Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # vLLM inference engine (v1 engine with auto-optimization)
  vllm:
    image: vllm/vllm-openai:latest
    command:
      - python
      - -m
      - vllm.entrypoints.openai.api_server
      - --model
      - Qwen/Qwen2.5-7B-Instruct
      - --host
      - 0.0.0.0
      - --port
      - '8000'
      - --dtype
      - float16
      - --gpu-memory-utilization
      - '0.9'
      # Note: v1 engine auto-handles chunked prefill and scheduling
      # Don't add --enable-chunked-prefill or --num-scheduler-steps
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - llm-network

  # FastAPI orchestration layer
  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    ports:
      - '8001:8001'
    environment:
      - VLLM_URL=http://vllm:8000
    depends_on:
      - vllm
    networks:
      - llm-network

networks:
  llm-network:
    driver: bridge
```

---

## Step 3: Create Orchestrator Dockerfile

Create `Dockerfile.orchestrator`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn httpx pydantic

COPY tool_orchestrator.py .

EXPOSE 8001

CMD ["uvicorn", "tool_orchestrator:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## Step 4: Deploy the Multi-Service Stack to AWS

### Upload Files to Your AWS EC2 Instance

```bash
# From your local Mac/PC
# Upload docker-compose.yml and tool_orchestrator.py to AWS
scp -i llm-server-key.pem docker-compose.yml tool_orchestrator.py \
    ubuntu@YOUR_INSTANCE_IP:~/

# Also upload Dockerfile.orchestrator
scp -i llm-server-key.pem Dockerfile.orchestrator ubuntu@YOUR_INSTANCE_IP:~/
```

### Start All Services on AWS

```bash
# SSH into your AWS EC2 instance
ssh -i llm-server-key.pem ubuntu@YOUR_INSTANCE_IP

# Start both vLLM and orchestrator
docker-compose up -d

# This will:
# 1. Pull vLLM image (~10GB)
# 2. Download Qwen 7B model (~14GB)
# 3. Build orchestrator image (~1GB)
# 4. Start both services in containers
```

### Check Logs on AWS

```bash
# On your AWS EC2 instance - watch both services start up
docker-compose logs -f

# Wait for these messages:
# vllm_1         | Application startup complete
# orchestrator_1 | Application startup complete
# orchestrator_1 | Uvicorn running on http://0.0.0.0:8001
```

### Verify Services (from Your Local Machine)

```bash
# From your Mac/PC
export AWS_IP=54.123.45.67  # Replace with your EC2 public IP

# Check vLLM health
curl http://${AWS_IP}:8000/health
# Response: {"status":"ok"}

# Check orchestrator health
curl http://${AWS_IP}:8001/v1/models
# Response: Should forward to vLLM and show model info
```

**Important:** Update your AWS Security Group to allow inbound traffic on port 8001 (orchestrator) in addition to port 8000 (vLLM).

---

## Step 5: Test Tool Calling from Your Local Machine

Now test the tool calling feature from your Mac/PC, calling the AWS orchestrator.

### Weather Tool Example

```bash
# From your Mac/PC
export AWS_IP=54.123.45.67  # Replace with your EC2 public IP

curl http://${AWS_IP}:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city name, e.g., San Francisco"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

**Expected Flow:**
1. Orchestrator receives request with tool definition
2. Calls vLLM: "What is the weather in San Francisco?"
3. vLLM responds with tool call: `{"tool": "get_current_weather", "arguments": {"location": "San Francisco"}}`
4. Orchestrator executes tool: `get_current_weather("San Francisco")`
5. Tool returns: `"The weather in San Francisco is 22°C and sunny."`
6. Orchestrator calls vLLM again with tool result
7. vLLM generates final response: `"The weather in San Francisco is 22°C and sunny."`

### Calculator Tool Example

```bash
# From your Mac/PC
curl http://${AWS_IP}:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Calculate 123 * 456"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate_math",
          "description": "Evaluate a mathematical expression",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g., 2+2"
              }
            },
            "required": ["expression"]
          }
        }
      }
    ]
  }'

# Expected response:
# "The result of 123 * 456 is 56088"
```

### Python SDK Example (from Your Mac)

```python
from openai import OpenAI

# Point to AWS orchestrator (NOT vLLM directly)
client = OpenAI(
    base_url="http://54.123.45.67:8001/v1",  # Replace with your EC2 IP
    api_key="dummy"
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=tools
)

print(response.choices[0].message.content)
```

---

## Step 6: Add Your Own Tools

### Custom Tool Example: Database Query

Add to `tool_orchestrator.py`:

```python
def query_database(table: str, filter_by: str) -> str:
    """Query your database - replace with real DB connection"""
    # Example: Connect to PostgreSQL
    # conn = psycopg2.connect(...)
    # cursor = conn.cursor()
    # cursor.execute(f"SELECT * FROM {table} WHERE {filter_by}")
    # results = cursor.fetchall()
    return f"Query results from {table}: [sample data]"

# Add to AVAILABLE_TOOLS
AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather,
    "calculate_math": calculate_math,
    "query_database": query_database,  # New tool!
}
```

Rebuild orchestrator:
```bash
docker-compose up -d --build orchestrator
```

---

## Step 7: Production Considerations

### Independent Scaling

**Scale Orchestrator (CPU):**
```yaml
# docker-compose.yml
orchestrator:
  deploy:
    replicas: 5  # Scale horizontally for more request handling
```

**Scale vLLM (GPU):**
```bash
# Add more GPU instances behind a load balancer
# Or use tensor parallelism for larger models
```

### Monitoring

```python
# Add to tool_orchestrator.py
from prometheus_client import Counter, Histogram

tool_calls = Counter('tool_calls_total', 'Total tool calls', ['tool_name'])
latency = Histogram('request_latency_seconds', 'Request latency')

@app.post("/v1/chat/completions")
async def chat_with_tools(request: ChatRequest):
    with latency.time():
        # ... existing code ...
        if tool_call:
            tool_calls.labels(tool_name=tool_name).inc()
```

### Security

```python
# Add authentication
from fastapi import Header, HTTPException

@app.post("/v1/chat/completions")
async def chat_with_tools(
    request: ChatRequest,
    api_key: str = Header(None, alias="X-API-Key")
):
    if api_key != os.getenv("VALID_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... existing code ...
```

---

## Step 8: Deploy to AWS ECS (Production)

For production deployments, use AWS ECS to deploy the two-tier architecture with independent scaling.

### Architecture on AWS ECS

```
Internet → ALB (Port 443) → Orchestrator Tasks (CPU) → vLLM Tasks (GPU)
                                   ↓                          ↓
                            Auto Scaling (3-10)        Single Task (g5.xlarge)
```

### Step-by-Step ECS Deployment

**1. Push Docker Images to ECR**

```bash
# From your Mac/PC
# Create ECR repositories
aws ecr create-repository --repository-name vllm-service
aws ecr create-repository --repository-name orchestrator-service

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag and push images
docker tag vllm/vllm-openai:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/vllm-service:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/vllm-service:latest

docker tag my-orchestrator:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/orchestrator-service:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/orchestrator-service:latest
```

**2. Create ECS Task Definitions**

**vLLM Task Definition (GPU):**
```json
{
  "family": "vllm-task",
  "requiresCompatibilities": ["EC2"],
  "networkMode": "bridge",
  "containerDefinitions": [
    {
      "name": "vllm",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/vllm-service:latest",
      "memory": 24576,
      "portMappings": [{"containerPort": 8000}],
      "resourceRequirements": [
        {"type": "GPU", "value": "1"}
      ]
    }
  ]
}
```

**Orchestrator Task Definition (CPU):**
```json
{
  "family": "orchestrator-task",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "orchestrator",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/orchestrator-service:latest",
      "portMappings": [{"containerPort": 8001}],
      "environment": [
        {"name": "VLLM_URL", "value": "http://vllm-service:8000"}
      ]
    }
  ]
}
```

**3. Create ECS Services**

**vLLM Service:**
- Cluster: Your ECS cluster with GPU instances (g5.xlarge)
- Task definition: vllm-task
- Desired count: 1 (GPU is expensive)
- Launch type: EC2 (for GPU support)

**Orchestrator Service:**
- Cluster: Same ECS cluster
- Task definition: orchestrator-task
- Desired count: 3-10
- Launch type: FARGATE (CPU only)
- Auto Scaling: Based on CPU utilization
- Load balancer: Application Load Balancer on port 443

**4. Configure Application Load Balancer**

- **Listener**: HTTPS (443) with ACM certificate
- **Target Group**: Orchestrator tasks on port 8001
- **Health Check**: `/v1/models` endpoint

**Full guide:** See [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md) for complete AWS ECS setup

### Alternative: AWS EKS (Kubernetes)

For Kubernetes on AWS EKS, see [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md) for full manifests with GPU node pools.

---

## Troubleshooting

### "Connection refused" from orchestrator to vLLM

**Problem:** Orchestrator can't reach vLLM service

**Solution:**
```yaml
# In docker-compose.yml, ensure both services use same network
networks:
  - llm-network

# Check orchestrator environment variable
environment:
  - VLLM_URL=http://vllm:8000  # Use service name, not localhost
```

### Tool call not detected

**Problem:** Model doesn't return tool call JSON

**Solution:** The current implementation uses regex to detect tool calls. For production, use models fine-tuned for tool calling:
- OpenAI: gpt-4-turbo
- Anthropic: claude-3-opus
- Open source: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct

### Slow response times

**Problem:** Two inference passes take too long

**Solution:**
```python
# Enable streaming responses
# Reduce max_tokens for initial pass
# Use prefix caching in vLLM for repeated system prompts
```

---

## Next Steps

**Completed Lab 9?** Explore advanced topics:

→ **[Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)** - Kubernetes autoscaling, multi-model serving

→ **[Real-World Case Studies](REAL_WORLD_DEPLOYMENT_BLOGS.md)** - Learn how Salesforce and others deploy tool calling

→ **Add RAG:** Integrate vector databases (Pinecone, Weaviate) into the orchestrator

→ **Add guardrails:** Implement content moderation (Azure Content Safety, Anthropic's Claude Moderation)

---

## Key Takeaways

✅ Two-tier architecture separates concerns (business logic vs inference)
✅ FastAPI scales horizontally on CPU, vLLM scales vertically on GPU
✅ Docker Compose simplifies multi-service development
✅ Tool calling requires two inference passes (detection + final response)
✅ Production systems at Anthropic/OpenAI/Google use this pattern

**You now have a production-ready LLM tool calling system!**

---

## References

- **vLLM v1 Blog:** [https://blog.vllm.ai/2025/01/27/v1-alpha-release.html](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- **vLLM Documentation:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **FastAPI Documentation:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Docker Compose Documentation:** [https://docs.docker.com/compose/](https://docs.docker.com/compose/)

**Last Updated:** October 2025 | **vLLM Version:** v0.11.1 (October 2025)
