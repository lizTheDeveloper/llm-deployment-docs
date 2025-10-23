# Lab 9: FastAPI Tool Calling with vLLM

Build a production two-tier architecture: FastAPI orchestration layer + vLLM inference engine for LLM tool calling.

**Time:** 45 minutes
**Prerequisites:** Completed [Lab 8](LAB_8_VLLM_DEPLOYMENT.md), Docker Compose, cloud GPU instance
**What You'll Build:** Multi-service deployment with tool calling (weather, calculator) and vLLM backend

---

## What You'll Learn

- Design two-tier LLM architectures (orchestration + inference)
- Implement tool calling with FastAPI
- Deploy multi-service applications with Docker Compose
- Separate business logic from inference
- Scale orchestration and inference layers independently

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

## Step 4: Deploy the Multi-Service Stack

### Start All Services

```bash
# Start both vLLM and orchestrator
docker-compose up -d

# This will:
# 1. Pull vLLM image (~10GB)
# 2. Download Qwen 7B model (~14GB)
# 3. Build orchestrator image (~1GB)
# 4. Start both services in containers
```

### Check Logs

```bash
# Watch both services start up
docker-compose logs -f

# Wait for these messages:
# vllm_1         | Application startup complete
# orchestrator_1 | Application startup complete
# orchestrator_1 | Uvicorn running on http://0.0.0.0:8001
```

### Verify Services

```bash
# Check vLLM health
curl http://localhost:8000/health
# Response: {"status":"ok"}

# Check orchestrator health
curl http://localhost:8001/v1/models
# Response: Should forward to vLLM and show model info
```

---

## Step 5: Test Tool Calling

### Weather Tool Example

```bash
curl http://localhost:8001/v1/chat/completions \
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
curl http://localhost:8001/v1/chat/completions \
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

### Python SDK Example

```python
from openai import OpenAI

# Point to orchestrator (NOT vLLM directly)
client = OpenAI(
    base_url="http://localhost:8001/v1",
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

## Step 8: Deploy to Cloud

### AWS ECS Deployment

Create two ECS services:

**vLLM Service (GPU tasks):**
- Instance type: g5.xlarge
- Task definition: vLLM container
- Port: 8000 (internal only)

**Orchestrator Service (CPU tasks):**
- Instance type: t3.medium
- Task definition: Orchestrator container
- Port: 8001 (public via load balancer)
- Scale: 3-10 replicas (autoscaling)

### Kubernetes Deployment

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  replicas: 1  # GPU is expensive, scale vertically
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1

---
# orchestrator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
spec:
  replicas: 5  # CPU is cheap, scale horizontally
  template:
    spec:
      containers:
      - name: orchestrator
        image: my-orchestrator:latest
```

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
