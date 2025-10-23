#!/bin/bash
# Quick start script to run MLX-LM natively on Mac (fastest option)

set -e

echo "🚀 MLX-LM Native Setup for Mac (M1/M2/M3)"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "mlx-env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv mlx-env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source mlx-env/bin/activate

# Install dependencies
echo "📥 Installing MLX-LM and dependencies..."
pip install --quiet --upgrade pip
pip install --quiet mlx mlx-lm fastapi uvicorn[standard] pydantic

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Starting MLX-LM OpenAI-compatible server..."
echo "   Model: mlx-community/Qwen2.5-3B-Instruct-4bit"
echo "   This will download the model on first run (~2.3GB)"
echo ""
echo "📡 Server will be available at:"
echo "   http://localhost:8000/v1/chat/completions"
echo "   http://localhost:8000/v1/models"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python mlx_api_server.py \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 0.0.0.0 \
  --port 8000

