# GCP GKE vLLM Deployment - Complete kubectl Commands

**Complete step-by-step guide with all kubectl ("cube control") commands for deploying vLLM on Google Kubernetes Engine**

## Prerequisites

```bash
# Install gcloud SDK (if not already installed)
# https://cloud.google.com/sdk/docs/install

# Install kubectl plugin
gcloud components install gke-gcloud-auth-plugin kubectl

# Login to GCP
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
```

## Step 1: Create GKE Autopilot Cluster

**What is Autopilot?** Fully managed Kubernetes - Google handles node provisioning, scaling, and upgrades.

```bash
# Create cluster (takes ~10-15 minutes)
gcloud container clusters create-auto vllm-cluster \
  --location=us-central1 \
  --release-channel=rapid \
  --project=YOUR_PROJECT_ID

# Output:
# Creating cluster vllm-cluster in us-central1...
# ....................................................done.
# Created [https://container.googleapis.com/v1/projects/YOUR_PROJECT/zones/us-central1/clusters/vllm-cluster].
# NAME          LOCATION     MASTER_VERSION  MACHINE_TYPE  NODE_VERSION  NUM_NODES  STATUS
# vllm-cluster  us-central1  1.34.0-gke...   ek-standard-8 1.34.0-gke... 3          RUNNING
```

## Step 2: Get kubectl Credentials

```bash
# Configure kubectl to use your cluster
gcloud container clusters get-credentials vllm-cluster --location=us-central1

# Output:
# Fetching cluster endpoint and auth data.
# kubeconfig entry generated for vllm-cluster.
```

### Verify kubectl Access

```bash
# Check cluster connection
kubectl get nodes

# Output shows initial nodes (GPU nodes will be added when you deploy):
# NAME                                    STATUS   ROLES    AGE   VERSION
# gk3-vllm-cluster-default-pool-xxxxx     Ready    <none>   5m    v1.34.0-gke.2201000
```

## Step 3: Create Hugging Face Secret

**Why needed:** Many LLMs (Llama, Qwen) require Hugging Face authentication to download models.

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Create Kubernetes secret
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=$HF_TOKEN

# Output:
# secret/hf-secret created

# Verify secret
kubectl get secrets
```

## Step 4: Create Deployment YAML

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
  type: LoadBalancer  # Creates external IP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-deployment
spec:
  replicas: 1  # Start with 1, scale with: kubectl scale deployment llama-deployment --replicas=4
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
            cpu: "8"           # GKE uses this to select machine type
            memory: "32Gi"
            ephemeral-storage: "40Gi"  # For model downloads
            nvidia.com/gpu: 1  # Request 1 GPU
          limits:
            cpu: "8"
            memory: "32Gi"
            ephemeral-storage: "40Gi"
            nvidia.com/gpu: 1
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4  # Force L4 GPU (24GB, cost-effective)
```

## Step 5: Deploy to Cluster

```bash
# Apply the deployment
kubectl apply -f llama-8b-deployment.yaml

# Output:
# service/llama-service created
# deployment.apps/llama-deployment created
```

### What Happens Now?

1. **GKE provisions GPU node** (~5-10 min) - Creates g2-standard-8 instance with L4 GPU
2. **Pulls vLLM image** (~2-3 min) - Downloads vllm/vllm-openai:v0.11.1 (6-8GB)
3. **Downloads model** (~3-5 min) - Llama 3.1 8B (~16GB from Hugging Face)
4. **Starts vLLM server** (~1-2 min) - Loads model into GPU memory
5. **Service ready** - Total time: ~15-20 minutes

## Step 6: Monitor Deployment

### Check Pod Status

```bash
# Watch pod creation
kubectl get pods -l app=llama-vllm -w

# Output progression:
# NAME                               READY   STATUS              RESTARTS   AGE
# llama-deployment-xxx-yyy           0/1     Pending             0          10s
# llama-deployment-xxx-yyy           0/1     ContainerCreating   0          5m
# llama-deployment-xxx-yyy           0/1     Running             0          8m
# llama-deployment-xxx-yyy           1/1     Running             0          15m  # READY!
```

### Check GPU Node Provisioning

```bash
# See nodes (GPU node appears after ~5 min)
kubectl get nodes

# Output:
# NAME                                   STATUS   ROLES    AGE   VERSION
# gk3-vllm-cluster-pool-2-xxxxx          Ready    <none>   6m    v1.34.0-gke.2201000  # <-- GPU node!
```

### View Logs (Troubleshooting)

```bash
# Stream logs
kubectl logs -f deployment/llama-deployment

# Output shows:
# INFO:     Started server process [1]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO     vllm.engine.arg_utils:  model='meta-llama/Meta-Llama-3.1-8B-Instruct'
# INFO     vllm.engine.llm_engine:  # GPU blocks: 1234, # CPU blocks: 512
```

### Describe Pod (Detailed Status)

```bash
# Get detailed pod info
kubectl describe pod -l app=llama-vllm

# Shows events:
#   Normal   Scheduled    5m    Successfully assigned default/llama-deployment-xxx to gk3-vllm-cluster-pool-2-xxx
#   Normal   Pulling      4m    Pulling image "vllm/vllm-openai:v0.11.1"
#   Normal   Pulled       2m    Successfully pulled image
#   Normal   Created      2m    Created container vllm-server
#   Normal   Started      2m    Started container vllm-server
```

## Step 7: Test the Deployment

### Option A: Port Forwarding (Quick Test)

```bash
# Forward port 8000 to localhost:8080
kubectl port-forward service/llama-service 8080:8000

# In another terminal, test:
curl http://localhost:8080/v1/models

# Output:
# {"object":"list","data":[{"id":"meta-llama/Meta-Llama-3.1-8B-Instruct","object":"model",...}]}
```

### Test Inference

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Option B: External IP (Production)

```bash
# Get external IP (takes 2-3 minutes to provision)
kubectl get service llama-service

# Output:
# NAME            TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)          AGE
# llama-service   LoadBalancer   34.118.226.54   35.232.123.45    8000:30802/TCP   5m

# Test with external IP:
curl http://35.232.123.45:8000/v1/models
```

## Essential kubectl Commands Reference

### Deployment Management

```bash
# List all deployments
kubectl get deployments

# Scale deployment (for 400 users → 4 replicas)
kubectl scale deployment llama-deployment --replicas=4

# Update deployment image
kubectl set image deployment/llama-deployment vllm-server=vllm/vllm-openai:v0.11.2

# Rollback deployment
kubectl rollout undo deployment/llama-deployment

# Check rollout status
kubectl rollout status deployment/llama-deployment
```

### Pod Management

```bash
# List pods
kubectl get pods
kubectl get pods -l app=llama-vllm  # Filter by label
kubectl get pods -o wide  # Show node assignments

# Describe pod
kubectl describe pod llama-deployment-xxx-yyy

# Get logs
kubectl logs llama-deployment-xxx-yyy
kubectl logs -f deployment/llama-deployment  # Follow logs
kubectl logs --tail=100 deployment/llama-deployment  # Last 100 lines

# Execute command in pod
kubectl exec -it llama-deployment-xxx-yyy -- /bin/bash
kubectl exec -it llama-deployment-xxx-yyy -- nvidia-smi  # Check GPU
```

### Service Management

```bash
# List services
kubectl get services
kubectl get svc  # Short form

# Describe service
kubectl describe service llama-service

# Delete service
kubectl delete service llama-service
```

### Resource Monitoring

```bash
# Get resource usage
kubectl top nodes  # Node CPU/memory
kubectl top pods   # Pod CPU/memory

# Watch resources
kubectl get pods -w  # Watch pods
kubectl get events --watch  # Watch cluster events
```

### Cleanup

```bash
# Delete deployment
kubectl delete deployment llama-deployment

# Delete service
kubectl delete service llama-service

# Delete everything in YAML file
kubectl delete -f llama-8b-deployment.yaml

# Delete secret
kubectl delete secret hf-secret

# Delete entire cluster (careful!)
gcloud container clusters delete vllm-cluster --location=us-central1
```

## Common Troubleshooting Commands

### Pod Won't Start

```bash
# Check pod status
kubectl get pods -l app=llama-vllm

# Get detailed events
kubectl describe pod llama-deployment-xxx-yyy

# Check logs for errors
kubectl logs llama-deployment-xxx-yyy --previous  # Previous crash logs
```

### GPU Not Available

```bash
# Check if GPU node exists
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4

# Describe GPU node
kubectl describe node gk3-vllm-cluster-pool-2-xxx | grep -A 10 "Capacity:"

# Should show:
#   nvidia.com/gpu:  1
```

### Image Pull Errors

```bash
# Check events
kubectl get events --field-selector involvedObject.name=llama-deployment-xxx

# Check if secret exists
kubectl get secret hf-secret -o yaml
```

### Service External IP Pending

```bash
# Check service
kubectl get service llama-service

# Describe for events
kubectl describe service llama-service

# If stuck, delete and recreate
kubectl delete service llama-service
kubectl apply -f llama-8b-deployment.yaml
```

## Cost Optimization Commands

### Auto-Scaling

```bash
# Install Horizontal Pod Autoscaler
kubectl autoscale deployment llama-deployment \
  --min=1 \
  --max=10 \
  --cpu-percent=80

# Check HPA status
kubectl get hpa
```

### Use Spot Instances (70% cheaper)

Modify deployment YAML:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
        cloud.google.com/gke-spot: "true"  # Add this line
```

## Summary: GKE vs AWS ECS

| Command Type | AWS ECS | GKE kubectl |
|--------------|---------|-------------|
| **Deploy** | `aws ecs create-service ...` | `kubectl apply -f deployment.yaml` |
| **Scale** | `aws ecs update-service --desired-count 4` | `kubectl scale deployment --replicas=4` |
| **Logs** | `aws logs tail ...` | `kubectl logs -f deployment/name` |
| **Status** | `aws ecs describe-tasks ...` | `kubectl get pods` |
| **Delete** | `aws ecs delete-service ...` | `kubectl delete deployment name` |

**Verdict:** kubectl is simpler and more intuitive than AWS CLI for container management.

---

**Tested and verified:** October 2025 on GKE Autopilot v1.34
