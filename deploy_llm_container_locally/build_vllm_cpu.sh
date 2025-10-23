#!/bin/bash
# Build vLLM CPU-only Docker image (not recommended for Mac)

set -e

echo "⚠️  WARNING: Building vLLM CPU-only Docker Image"
echo "=================================================="
echo ""
echo "This will:"
echo "  - Take 20-30 minutes to build"
echo "  - Use 4-8GB of disk space"
echo "  - Result in 1-5 tokens/second inference speed"
echo ""
echo "❌ NOT RECOMMENDED for Mac users!"
echo ""
echo "✅ BETTER OPTION: Use MLX-LM natively (20-60 tok/s)"
echo "   Just run: ./run_mlx_native.sh"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Build cancelled. Run ./run_mlx_native.sh instead for better performance."
    exit 1
fi

echo "🔨 Starting vLLM CPU build..."
echo "This will take approximately 20-30 minutes..."
echo ""

# Build with progress
docker build \
    --progress=plain \
    -t vllm-cpu:latest \
    -f Dockerfile.vllm-cpu \
    .

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the server:"
echo "  docker run -d -p 8000:8000 --name vllm-cpu vllm-cpu:latest"
echo ""
echo "To test:"
echo "  python test_api.py"
echo ""
echo "⚠️  Remember: This will be MUCH slower than native MLX-LM"

