#!/bin/bash
# GCP GKE vLLM Quick Test Script
# Equivalent to AWS ECS small deployment (400 users, Llama 3.1 8B)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GCP GKE vLLM Test Deployment ===${NC}"

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found. Install with: gcloud components install kubectl${NC}"
    exit 1
fi

# Set variables
read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter your Hugging Face token: " HF_TOKEN
export REGION="us-central1"  # L4 GPUs available
export CLUSTER_NAME="vllm-test-cluster"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Cluster: $CLUSTER_NAME"
echo "  GPU: NVIDIA L4 (24GB)"
echo "  Model: Llama 3.1 8B"

# Set project
echo -e "${GREEN}Step 1: Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Create GKE Autopilot cluster
echo -e "${GREEN}Step 2: Creating GKE Autopilot cluster (takes ~5-10 min)...${NC}"
if gcloud container clusters describe $CLUSTER_NAME --location=$REGION &> /dev/null; then
    echo -e "${YELLOW}Cluster already exists, skipping creation${NC}"
else
    gcloud container clusters create-auto $CLUSTER_NAME \
      --location=$REGION \
      --release-channel=rapid \
      --cluster-version=1.31 || {
        echo -e "${RED}Failed to create cluster. Check quota and permissions.${NC}"
        exit 1
    }
fi

# Get credentials
echo -e "${GREEN}Step 3: Getting cluster credentials...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --location=$REGION

# Create Hugging Face secret
echo -e "${GREEN}Step 4: Creating Hugging Face secret...${NC}"
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=${HF_TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -

# Create deployment YAML
echo -e "${GREEN}Step 5: Creating deployment configuration...${NC}"
cat > /tmp/llama-8b-deployment.yaml <<EOF
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
  replicas: 1
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
            nvidia.com/gpu: 1
          limits:
            cpu: "8"
            memory: "32Gi"
            ephemeral-storage: "40Gi"
            nvidia.com/gpu: 1
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
EOF

# Deploy
echo -e "${GREEN}Step 6: Deploying vLLM (this takes 5-10 min for GPU node provisioning)...${NC}"
kubectl apply -f /tmp/llama-8b-deployment.yaml

# Wait for deployment
echo -e "${YELLOW}Waiting for pod to be ready...${NC}"
kubectl wait --for=condition=Available --timeout=1800s deployment/llama-deployment || {
    echo -e "${RED}Deployment failed. Checking logs...${NC}"
    kubectl get pods
    kubectl logs -l app=llama-vllm --tail=50
    exit 1
}

# Get service details
echo -e "${GREEN}Step 7: Deployment successful!${NC}"
kubectl get pods -l app=llama-vllm
kubectl get service llama-service

# Test inference
echo -e "${GREEN}Step 8: Testing inference...${NC}"
echo -e "${YELLOW}Starting port-forward (Ctrl+C to stop)...${NC}"
kubectl port-forward service/llama-service 8080:8000 &
PF_PID=$!
sleep 5

# Test API
echo -e "${GREEN}Sending test request...${NC}"
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is machine learning? Answer in one sentence."}],
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq

kill $PF_PID

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Access your vLLM server with:"
echo "  kubectl port-forward service/llama-service 8080:8000"
echo "  curl http://localhost:8080/v1/models"
echo ""
echo "Scale to 4 replicas for 400 users:"
echo "  kubectl scale deployment llama-deployment --replicas=4"
echo ""
echo "View logs:"
echo "  kubectl logs -f deployment/llama-deployment"
echo ""
echo "Cleanup (delete everything):"
echo "  gcloud container clusters delete $CLUSTER_NAME --location=$REGION --quiet"
echo ""
echo -e "${YELLOW}Cost estimate: ~$0.80-1.20/hour per replica (L4 GPU)${NC}"
