#!/bin/bash
# Quick setup for Docker LLM demo using llama.cpp

set -e

echo "🐳 Setting up Docker LLM Demo"
echo "=============================="
echo ""

# Create models directory
mkdir -p models

# Check if model exists
if [ ! -f "models/qwen-3b-q4.gguf" ]; then
    echo "📥 Downloading model (Qwen 2.5 3B, Q4 quantized, ~2GB)..."
    echo "This will take 2-5 minutes depending on your connection..."
    echo ""
    
    curl -L -o models/qwen-3b-q4.gguf \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    echo ""
    echo "✅ Model downloaded!"
else
    echo "✅ Model already exists"
fi

echo ""
echo "🐳 Starting Docker container..."
echo ""

# Stop MLX server if running (to free port 8000)
echo "Stopping MLX server (if running)..."
pkill -f mlx_api_server 2>/dev/null || true

# Start container
docker-compose up -d

echo ""
echo "⏳ Waiting for server to start (30 seconds)..."
sleep 30

# Test the API
echo ""
echo "🧪 Testing API..."
curl -s http://localhost:8000/v1/models | python3 -m json.tool || echo "Waiting for model to load..."

echo ""
echo "================================"
echo "✅ Docker Demo Ready!"
echo "================================"
echo ""
echo "📡 API Endpoints:"
echo "   http://localhost:8000/v1/chat/completions"
echo "   http://localhost:8000/v1/completions"
echo "   http://localhost:8000/v1/models"
echo ""
echo "🧪 Test it:"
echo "   python3 test_api.py"
echo ""
echo "📊 Benchmark it:"
echo "   python3 benchmark.py 20 50"
echo ""
echo "🛑 Stop it:"
echo "   docker-compose down"
echo ""
echo "📋 View logs:"
echo "   docker logs llm-demo -f"
echo ""

