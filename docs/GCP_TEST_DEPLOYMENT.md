# GCP GKE Test Deployment - Small Scale (Llama 3.1 8B)

**Equivalent to AWS:** 400 concurrent users, ~3,000 tok/s required
**GCP Solution:** GKE Autopilot with L4 GPUs (24GB VRAM)

## GCP vs AWS Equivalents

| AWS | GCP |
|-----|-----|
| ECS (Elastic Container Service) | GKE (Google Kubernetes Engine) |
| Deep Learning AMI | Deep Learning VM Image OR Container-Optimized OS |
| p4d.24xlarge (8× A100 40GB) | a2-highgpu-8g (8× A100 40GB) |
| g5.12xlarge (4× A10G 24GB) | **g2-standard-8 (1× L4 24GB)** ← Cost-effective for small scale |

## Prerequisites

```bash
# Set up GCP project
export PROJECT_ID="your-project-id"
export REGION="us-central1"  # L4 GPUs available here
export CLUSTER_NAME="vllm-test-cluster"
export HF_TOKEN="your_hugging_face_token"

gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
```

## Step 1: Create GKE Autopilot Cluster

**Why Autopilot?** Similar to AWS ECS Fargate - fully managed, auto-scaling

```bash
# Create cluster (takes ~5-10 minutes)
gcloud container clusters create-auto $CLUSTER_NAME \
  --location=$REGION \
  --release-channel=rapid \
  --cluster-version=1.31

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --location=$REGION

# Verify cluster
kubectl get nodes
```

**Source:** https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-llama-gpus-vllm

## Step 2: Create Hugging Face Secret

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=${HF_TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Step 3: Deploy vLLM with Llama 3.1 8B

Create `llama-8b-deployment.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-service
spec:
  selector:
    app: llama-vllm
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-deployment
spec:
  replicas: 1  # Start with 1, can scale to 4 for 400 users
  selector:
    matchLabels:
      app: llama-vllm
  template:
    metadata:
      labels:
        app: llama-vllm
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:v0.11.1
        command:
        - python3
        - -m
        - vllm.entrypoints.openai.api_server
        - --model=meta-llama/Meta-Llama-3.1-8B-Instruct
        - --tensor-parallel-size=1
        - --gpu-memory-utilization=0.9
        - --dtype=float16
        - --port=8000
        ports:
        - containerPort: 8000
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: hf_api_token
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
            ephemeral-storage: "40Gi"
            nvidia.com/gpu: 1  # Request 1 L4 GPU
          limits:
            cpu: "8"
            memory: "32Gi"
            ephemeral-storage: "40Gi"
            nvidia.com/gpu: 1
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4  # L4 GPU (24GB)
```

Deploy:

```bash
kubectl apply -f llama-8b-deployment.yaml

# Watch deployment (takes 5-10 minutes for GPU node provisioning)
kubectl get pods -w

# Wait for ready
kubectl wait --for=condition=Available --timeout=1800s deployment/llama-deployment
```

**Key Differences from AWS:**
- Uses `nodeSelector` instead of `resourceRequirements`
- GKE Autopilot auto-provisions the right GPU machine type (g2-standard-8 with L4)
- No need to pre-create node pools in Autopilot mode

## Step 4: Test Deployment

```bash
# Get external IP (may take a minute)
kubectl get service llama-service

# Or use port-forward for testing
kubectl port-forward service/llama-service 8080:8000

# Test inference
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Expected response time: ~2 seconds for first token, ~793 tok/s throughput (based on benchmarks)

## Step 5: Scale for 400 Concurrent Users

To handle ~3,000 tok/s requirement:

```bash
# Scale to 4 replicas (4× L4 GPUs = ~3,200 tok/s)
kubectl scale deployment llama-deployment --replicas=4

# Verify scaling
kubectl get pods -l app=llama-vllm

# GKE will automatically provision 4 nodes with L4 GPUs
```

## GCP Machine Types Used

Based on resource requests, GKE Autopilot will provision:

| Component | Machine Type | GPUs | vCPU | RAM | Cost/Hour* |
|-----------|--------------|------|------|-----|-----------|
| **Small (1 replica)** | g2-standard-8 | 1× L4 (24GB) | 8 | 32GB | ~$0.80-1.20 |
| **400 Users (4 replicas)** | 4× g2-standard-8 | 4× L4 (24GB) | 32 | 128GB | ~$3.20-4.80 |

*Approximate pricing - check current GCP pricing

**Comparison to AWS:**
- AWS g5.12xlarge (4× A10G): ~$10.18/hour
- GCP g2-standard-8 (4× single L4 nodes): ~$3.20-4.80/hour
- **GCP is ~50% cheaper for this small scale deployment**

## Alternative: Deep Learning VM (Non-Container)

If you don't want to use GKE, you can use a standalone VM:

```bash
# Create Deep Learning VM with L4 GPU
gcloud compute instances create vllm-vm \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-l4,count=1" \
  --metadata="install-nvidia-driver=True"

# SSH into VM
gcloud compute ssh vllm-vm --zone=us-central1-a

# On the VM, run vLLM in Docker
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:v0.11.1 \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype float16
```

**Image Family:** `common-cu121-debian-11-py310`
- **Equivalent to:** AWS Deep Learning AMI
- **Includes:** CUDA 12.1, Python 3.10, NVIDIA drivers
- **Source:** https://cloud.google.com/deep-learning-vm/docs/cli

## Monitoring and Logs

```bash
# View pod logs
kubectl logs -f deployment/llama-deployment

# Check GPU utilization (requires nvidia-smi in container)
kubectl exec -it deployment/llama-deployment -- nvidia-smi

# View vLLM metrics
kubectl port-forward service/llama-service 8080:8000
curl http://localhost:8080/metrics
```

## Cleanup

```bash
# Delete deployment
kubectl delete -f llama-8b-deployment.yaml

# Delete cluster (or it keeps charging you!)
gcloud container clusters delete $CLUSTER_NAME --location=$REGION --quiet
```

## Summary: GCP vs AWS for Small Deployment

| Aspect | AWS ECS | GCP GKE |
|--------|---------|---------|
| **Service** | ECS Tasks | GKE Pods |
| **Autoscaling** | Manual ASG setup | Built-in HPA |
| **GPU Instance** | g5.12xlarge (4× A10G) | g2-standard-8 (1× L4) |
| **Cost (400 users)** | ~$10.18/hr | ~$3.20-4.80/hr |
| **Setup Complexity** | Moderate (task definitions) | Lower (YAML + auto-provisioning) |
| **Image/AMI** | Deep Learning AMI | Deep Learning VM Image |
| **Container Runtime** | Docker on ECS | CRI on GKE |

**Verdict:** GCP GKE Autopilot is **easier and cheaper** for small deployments due to auto-provisioning and L4 GPU pricing.
