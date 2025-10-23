# Enterprise-Scale LLM Deployment Guide

**Target Audience:** Production deployments at Salesforce scale
**Last Updated:** January 2025 (vLLM v1 engine)

This guide covers deploying multiple models simultaneously, large parameter models, and multi-tenant LoRA serving for enterprise environments.

---

## Table of Contents

1. [Multi-Model Serving (Multiple 7B Models)](#multi-model-serving)
2. [70B Model Deployment](#70b-model-deployment)
3. [100GB+ Model Deployment (405B Parameters)](#405b-model-deployment)
4. [LoRA Multi-Tenancy for Customer-Specific Models](#lora-multi-tenancy)
5. [Production Architecture Patterns](#production-architecture)
6. [Cost Optimization Strategies](#cost-optimization)

---

## Multi-Model Serving (Multiple 7B Models) {#multi-model-serving}

### Use Case
Deploying multiple distinct 7B parameter models (e.g., Llama-3.1-8B-Instruct, Mistral-7B-Instruct, Qwen-7B) to serve different product lines or customer segments.

### Hardware Requirements per 7B Model

| Quantization | VRAM Required | Recommended GPU | Throughput |
|--------------|---------------|-----------------|------------|
| FP16 (best quality) | 16GB minimum | A10G (24GB), L4 (24GB) | ~30-50 tok/s |
| INT8 | 8GB | T4 (16GB), A10G | ~40-60 tok/s |
| INT4 (AWQ/GPTQ) | 4-6GB | T4 | ~50-70 tok/s |

**Recommended for production:** A10G (24GB) with FP16 for quality, or L4 (24GB) for cost efficiency.

### vLLM Production Stack Configuration

**Option 1: Single vLLM instance with multiple models** (simple, lower throughput)

```yaml
# vllm-multimodel.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-router
spec:
  selector:
    app: vllm
  ports:
    - port: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-8b
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - meta-llama/Meta-Llama-3.1-8B-Instruct
          - --tensor-parallel-size
          - "1"
          - --gpu-memory-utilization
          - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 24Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-mistral-7b
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - mistralai/Mistral-7B-Instruct-v0.2
          - --tensor-parallel-size
          - "1"
          - --gpu-memory-utilization
          - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 24Gi
```

**Option 2: Model-aware routing with KServe** (enterprise-grade)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: multi-model-serving
spec:
  predictor:
    vllm:
      servingEngineSpec:
        - modelSpec:
            modelName: llama-3.1-8b
            modelUrl: s3://models/llama-3.1-8b
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: 24Gi
            limits:
              nvidia.com/gpu: 1
              memory: 24Gi
        - modelSpec:
            modelName: mistral-7b
            modelUrl: s3://models/mistral-7b
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: 24Gi
```

### Docker Compose for Local/Dev Testing

```yaml
version: '3.8'

services:
  vllm-llama:
    image: vllm/vllm-openai:latest
    command:
      - --model
      - meta-llama/Meta-Llama-3.1-8B-Instruct
      - --port
      - "8000"
      - --gpu-memory-utilization
      - "0.9"
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vllm-mistral:
    image: vllm/vllm-openai:latest
    command:
      - --model
      - mistralai/Mistral-7B-Instruct-v0.2
      - --port
      - "8001"
      - --gpu-memory-utilization
      - "0.9"
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  router:
    image: nginx:alpine
    volumes:
      - ./nginx-multimodel.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - vllm-llama
      - vllm-mistral
```

### Performance Characteristics (vLLM v1 Engine)

- **v1 vs v0:** 1.7x average speedup (January 2025)
- **Continuous batching:** Automatically batches concurrent requests
- **Prefix caching:** Zero-overhead caching of repeated prompts
- **Throughput:** ~50-100 tokens/sec per 7B model on A10G

---

## 70B Model Deployment {#70b-model-deployment}

### Hardware Requirements

**Problem:** 70B models require 130-140GB VRAM (cannot fit on single GPU)

**Solution:** Tensor parallelism across multiple GPUs

| Configuration | Total VRAM | Cost/Hour (AWS) | Use Case |
|---------------|------------|-----------------|----------|
| 4× A100 40GB | 160GB | ~$13-16 | Production (p4d.24xlarge) |
| 4× A6000 48GB | 192GB | ~$8-12 | Cost-effective alternative |
| 2× A100 80GB | 160GB | ~$10-13 | Dual-GPU (p4de.24xlarge) |
| 2× A100 80GB + AWQ 4-bit | 70GB (35GB/GPU) | ~$10-13 | Memory-optimized |

**Recommended:** 4× A100 40GB for production stability, or 2× A100 80GB with AWQ quantization for cost efficiency.

### Tensor Parallelism Configuration

**Single-node deployment (4 GPUs):**

```bash
# Start vLLM server with tensor parallelism
vllm serve \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --dtype float16
```

**Kubernetes deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
spec:
  replicas: 1  # Start with 1, scale vertically not horizontally
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - meta-llama/Llama-2-70b-chat-hf
          - --tensor-parallel-size
          - "4"
          - --gpu-memory-utilization
          - "0.9"
          - --dtype
          - float16
        resources:
          limits:
            nvidia.com/gpu: 4  # Request 4 GPUs
            memory: 160Gi
        env:
        - name: NCCL_DEBUG
          value: "INFO"  # For debugging multi-GPU communication
```

### Performance Tuning

**Key metrics to monitor:**

```python
# vLLM exposes Prometheus metrics at /metrics
# Key metrics for autoscaling:
- vllm:num_requests_running        # Current request queue
- vllm:gpu_cache_usage_perc        # KV cache utilization
- vllm:time_to_first_token_seconds # TTFT latency
- vllm:time_per_output_token_seconds # TPOT throughput
```

**Tensor Parallelism Impact:**

- **TP=1 → TP=2:** KV Cache blocks increase 13.9x, allows 3.9x more token throughput
- **TP=4:** Best for 70B models (balances latency vs throughput)
- **TP=8:** Overkill for 70B, use for 180B+ models

**Quantization Options:**

```bash
# AWQ 4-bit quantization (reduces memory by 4x)
vllm serve \
  --model TheBloke/Llama-2-70B-chat-AWQ \
  --quantization awq \
  --tensor-parallel-size 2  # Now fits on 2× A100 80GB!
  --dtype half
```

---

## 100GB+ Model Deployment (405B Parameters) {#405b-model-deployment}

### Use Case
Meta Llama 3.1 405B, GPT-4 scale models (400B+ parameters)

### Hardware Requirements

| Configuration | Total VRAM | Memory Type | Cost/Hour (AWS) |
|---------------|------------|-------------|-----------------|
| FP16 (full precision) | 810-972GB | - | Not feasible |
| FP8 (H100/H200) | 405-486GB | Recommended | ~$32-40 (8×H100) |
| INT4 (AWQ/GPTQ) | 203-243GB | Acceptable | ~$24-32 (8×A100 80GB) |

**FP8 Advantages (H100/H200 only):**
- 2x memory reduction vs FP16
- Minimal quality degradation (<1% on benchmarks)
- Hardware-accelerated on Hopper architecture
- **Recommended for 405B production deployments**

### Pipeline Parallelism (Multi-Node)

**Why Pipeline Parallelism?**
- Tensor parallelism alone requires expensive all-reduce operations (InfiniBand)
- Pipeline parallelism uses cheaper point-to-point communication
- **6.6x better performance** without InfiniBand (2-way pipeline + 8-way tensor vs 16-way tensor)

**Configuration:**

```bash
# 405B model across 2 nodes, each with 8 GPUs
vllm serve \
  --model meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dtype float8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

**How it works:**
- **Tensor Parallelism (TP=8):** Splits each layer horizontally across 8 GPUs within a node
- **Pipeline Parallelism (PP=2):** Splits layers vertically across 2 nodes
  - Node 1: Layers 1-40
  - Node 2: Layers 41-80

**Kubernetes Multi-Node Deployment:**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm-llama-405b
spec:
  serviceName: vllm-405b
  replicas: 2  # 2 nodes for pipeline parallelism
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: vllm-405b
            topologyKey: kubernetes.io/hostname  # Force different nodes
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - meta-llama/Meta-Llama-3.1-405B-Instruct
          - --tensor-parallel-size
          - "8"
          - --pipeline-parallel-size
          - "2"
          - --dtype
          - float8  # Requires H100/H200
        resources:
          limits:
            nvidia.com/gpu: 8  # 8 GPUs per node
            memory: 640Gi
        env:
        - name: NCCL_IB_DISABLE
          value: "1"  # Disable InfiniBand if not available
        - name: NCCL_P2P_LEVEL
          value: "NVL"  # Use NVLink for intra-node communication
```

### AWS Instance Recommendations

| Instance Type | GPUs | VRAM/GPU | Total VRAM | Best For |
|---------------|------|----------|------------|----------|
| p5.48xlarge | 8× H100 | 80GB | 640GB | FP8 405B (single node) |
| p4de.24xlarge | 8× A100 | 80GB | 640GB | INT4 405B (single node) |
| 2× p5.48xlarge | 16× H100 | 80GB | 1280GB | FP8 405B with headroom |

### Performance Expectations

- **Throughput:** 10-30 tokens/sec (depends on batch size)
- **TTFT (Time to First Token):** 1-3 seconds
- **TPOT (Time Per Output Token):** 30-100ms
- **Max concurrent requests:** 10-50 (limited by KV cache memory)

---

## LoRA Multi-Tenancy for Customer-Specific Models {#lora-multi-tenancy}

### Use Case (Salesforce Context)
- Single base model deployment (e.g., Llama-3.1-70B)
- Multiple customers load/unload their own LoRA adapters
- No downtime or redeployment needed
- Cost-efficient serving of thousands of customer-specific models

### S-LoRA Architecture

**Performance Characteristics:**
- Serve **2,000+ adapters** simultaneously
- **4x throughput** vs vLLM-packed (for small number of adapters)
- **30x throughput** vs PEFT library
- Dynamic adapter loading/unloading

**How it works:**
1. Base model loaded once in GPU memory
2. LoRA adapters stored in CPU memory pool
3. Active adapters paged into GPU on-demand
4. Batched requests use different adapters simultaneously

### vLLM LoRA Multi-Tenancy Setup

**Server configuration:**

```bash
# Start vLLM with LoRA support
vllm serve \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --enable-lora \
  --max-loras 100 \
  --max-lora-rank 64 \
  --lora-dtype float16 \
  --tensor-parallel-size 4
```

**Client requests specify LoRA adapter:**

```python
import openai

client = openai.OpenAI(
    base_url="http://vllm-server:8000/v1",
    api_key="dummy"
)

# Customer A request
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Summarize this contract"}],
    extra_body={
        "lora_name": "customer-a-legal-adapter"  # Load Customer A's LoRA
    }
)

# Customer B request (different adapter)
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Generate sales email"}],
    extra_body={
        "lora_name": "customer-b-sales-adapter"  # Load Customer B's LoRA
    }
)
```

### Kubernetes Deployment with LoRA Support

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lora-adapters
data:
  adapters.json: |
    {
      "customer-a-legal-adapter": "s3://lora-adapters/customer-a/legal-v1",
      "customer-b-sales-adapter": "s3://lora-adapters/customer-b/sales-v2",
      "customer-c-support-adapter": "s3://lora-adapters/customer-c/support-v1"
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-lora-serving
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - meta-llama/Meta-Llama-3.1-70B-Instruct
          - --enable-lora
          - --max-loras
          - "100"
          - --max-lora-rank
          - "64"
          - --tensor-parallel-size
          - "4"
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: 200Gi  # Extra memory for LoRA adapter pool
        volumeMounts:
        - name: lora-cache
          mountPath: /lora-cache
        - name: adapter-config
          mountPath: /config
      volumes:
      - name: lora-cache
        emptyDir:
          sizeLimit: 50Gi  # Cache for frequently-used adapters
      - name: adapter-config
        configMap:
          name: lora-adapters
```

### Punica Multi-LoRA Batching

**Alternative to S-LoRA:** Punica framework (integrated into vLLM)

```python
# Punica enables batching requests with different LoRA adapters
# Example: Single batch with 3 different customer adapters

batch_requests = [
    {"prompt": "Legal query", "lora": "customer-a-legal"},
    {"prompt": "Sales query", "lora": "customer-b-sales"},
    {"prompt": "Support query", "lora": "customer-c-support"},
]

# vLLM processes all 3 in a single batch!
# Base model forward pass shared, LoRA weights applied per-request
```

**Performance:**
- **12x throughput** vs sequential processing of different adapters
- Minimal latency overhead (5-10% vs single adapter)

### Adapter Management API

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

# Register new customer adapter
@app.post("/adapters/register")
async def register_adapter(customer_id: str, s3_path: str):
    # Download adapter from S3 to local cache
    # Update vLLM adapter registry
    return {"status": "registered", "customer_id": customer_id}

# Unload unused adapter (free memory)
@app.post("/adapters/unload")
async def unload_adapter(customer_id: str):
    # Signal vLLM to remove from active pool
    return {"status": "unloaded"}

# List active adapters
@app.get("/adapters/active")
async def list_active_adapters():
    # Query vLLM metrics
    return {"active_adapters": [...]}
```

---

## Production Architecture Patterns {#production-architecture}

### Pattern 1: Two-Tier (FastAPI + vLLM)

**When to use:**
- Need custom business logic (tool calling, RAG, guardrails)
- Multiple models behind unified API
- Complex routing or preprocessing

```
Client → FastAPI (Orchestration) → vLLM (Inference)
         ↓
    Tool execution, RAG, caching
```

**Scaling:**
- FastAPI: Horizontal (CPU replicas)
- vLLM: Vertical (more GPUs per replica)

### Pattern 2: Direct vLLM (OpenAI-Compatible)

**When to use:**
- Simple inference workloads
- OpenAI drop-in replacement
- Minimal custom logic

```
Client → vLLM (OpenAI API) → GPU Inference
```

**Scaling:**
- vLLM: Horizontal (multiple replicas) + Vertical (tensor parallelism)

### Pattern 3: Multi-Tier with LoRA (Enterprise)

**When to use:**
- Multi-tenant deployments (Salesforce scale)
- Customer-specific fine-tuning
- Dynamic adapter loading

```
Client → API Gateway (Tenant routing) → FastAPI (Adapter management) → vLLM (Base + LoRA)
                                          ↓
                                    S3 (LoRA adapters)
```

### Pattern 4: Pipeline Parallelism (Massive Models)

**When to use:**
- 100GB+ models (405B parameters)
- Multi-node deployments
- Cost-optimized infrastructure

```
Client → Load Balancer → vLLM Controller → GPU Node 1 (Layers 1-40, TP=8)
                                          → GPU Node 2 (Layers 41-80, TP=8)
```

---

## Cost Optimization Strategies {#cost-optimization}

### 1. Right-Sizing GPU Instances

| Workload | Recommended GPU | Why |
|----------|-----------------|-----|
| 7B models (dev/test) | T4 (16GB) | $0.35/hour, sufficient for INT4 |
| 7B models (production) | A10G (24GB) | $1.01/hour, FP16 quality |
| 13B models | L4 (24GB) | $0.80/hour, best price/performance |
| 70B models | 4× A100 40GB | $13/hour, proven reliability |
| 405B models | 8× H100 80GB | $32/hour, FP8 support |

### 2. Quantization Trade-offs

| Method | Memory Reduction | Quality Loss | Speedup |
|--------|------------------|--------------|---------|
| FP8 (H100 only) | 2x | <1% | 1.5-2x |
| AWQ 4-bit | 4x | 2-3% | 1.2-1.5x |
| GPTQ 4-bit | 4x | 3-5% | 1.1-1.3x |
| INT8 | 2x | 1-2% | 1.3-1.6x |

**Recommendation:** Use FP8 on H100 for 70B+ models, AWQ for 7B-70B.

### 3. Autoscaling Strategies

**Metric-based autoscaling (Kubernetes HPA):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-7b
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_num_requests_running  # Custom metric from Prometheus
      target:
        type: AverageValue
        averageValue: "5"  # Scale up when >5 requests queued per pod
```

**Keda autoscaling (event-driven):**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-7b
  minReplicaCount: 1
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_request_queue_depth
      threshold: '10'
      query: |
        sum(vllm:num_requests_waiting)
```

### 4. Spot Instance Strategies

**AWS Spot Instances (70% cost reduction):**

```yaml
# Kubernetes node group with spot instances
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: gpu-spot
spec:
  requirements:
  - key: karpenter.sh/capacity-type
    operator: In
    values: ["spot"]
  - key: node.kubernetes.io/instance-type
    operator: In
    values: ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"]
  limits:
    resources:
      nvidia.com/gpu: 100
  labels:
    workload: inference
  taints:
  - key: nvidia.com/gpu
    effect: NoSchedule
```

**Spot interruption handling:**
- Use 2+ replicas for high availability
- Graceful shutdown hooks (save in-flight requests)
- Fallback to on-demand instances

### 5. Request Batching

**vLLM v1 auto-batching:**
- Continuous batching enabled by default
- Dynamically batches concurrent requests
- No configuration needed

**Optimal batch sizes:**
- **7B models:** 32-64 concurrent requests
- **70B models:** 8-16 concurrent requests
- **405B models:** 2-8 concurrent requests

### 6. Prefix Caching (vLLM v1)

**Use case:** Repeated system prompts, RAG contexts

```python
# Example: RAG with cached context
response = client.chat.completions.create(
    model="llama-3.1-70b",
    messages=[
        {"role": "system", "content": long_rag_context},  # Cached!
        {"role": "user", "content": "What is the answer?"}
    ]
)
# Second request with same context → 0ms context processing
```

**Savings:**
- 50-90% reduction in TTFT for repeated contexts
- Zero overhead (automatic in vLLM v1)

---

## Monitoring and Observability

### vLLM Prometheus Metrics

```yaml
# Key metrics to monitor
- vllm:num_requests_running        # Active inference requests
- vllm:num_requests_waiting        # Queue depth (autoscaling trigger)
- vllm:gpu_cache_usage_perc        # KV cache utilization
- vllm:time_to_first_token_seconds # Latency (TTFT)
- vllm:time_per_output_token_seconds # Throughput (TPOT)
- vllm:prompt_tokens_total         # Input token usage
- vllm:generation_tokens_total     # Output token usage
```

### Grafana Dashboard Example

```yaml
# Grafana dashboard for vLLM
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-dashboard
data:
  dashboard.json: |
    {
      "panels": [
        {
          "title": "Request Queue Depth",
          "targets": [{"expr": "vllm:num_requests_waiting"}]
        },
        {
          "title": "GPU Memory Usage",
          "targets": [{"expr": "vllm:gpu_cache_usage_perc"}]
        },
        {
          "title": "Throughput (tokens/sec)",
          "targets": [{"expr": "rate(vllm:generation_tokens_total[1m])"}]
        }
      ]
    }
```

---

## Quick Reference: Common Configurations

### Multi-7B Models (Cost-Optimized)
```bash
# 3× 7B models on A10G instances
vllm serve --model llama-3.1-8b --gpu-memory-utilization 0.9 &
vllm serve --model mistral-7b --port 8001 --gpu-memory-utilization 0.9 &
vllm serve --model qwen-7b --port 8002 --gpu-memory-utilization 0.9 &
```

### 70B Model (Production)
```bash
# 4× A100 40GB with tensor parallelism
vllm serve \
  --model llama-2-70b-chat \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --dtype float16
```

### 405B Model (Multi-Node)
```bash
# 2 nodes × 8 H100 GPUs with pipeline + tensor parallelism
vllm serve \
  --model llama-3.1-405b \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dtype float8 \
  --gpu-memory-utilization 0.9
```

### LoRA Multi-Tenancy (Salesforce Scale)
```bash
# Single 70B base + 100 customer LoRA adapters
vllm serve \
  --model llama-3.1-70b \
  --enable-lora \
  --max-loras 100 \
  --max-lora-rank 64 \
  --tensor-parallel-size 4
```

---

## Additional Resources

- **vLLM Documentation:** https://docs.vllm.ai/
- **vLLM v1 Announcement:** https://blog.vllm.ai/2025/01/13/v1.html
- **S-LoRA Paper:** https://arxiv.org/abs/2311.03285
- **Punica Framework:** https://github.com/punica-ai/punica
- **AWS GPU Instances:** https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing
- **NVIDIA GPU Memory Calculator:** https://resources.nvidia.com/en-us-llm

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Maintained by:** Mastering LLM Deployment Course
