#!/bin/bash
# Build vLLM CPU-only Docker image with fixes

set -e

echo "⚠️  Building vLLM CPU Docker Image (Fixed Version)"
echo "=================================================="
echo ""
echo "This will:"
echo "  - Use Ubuntu 22.04 base with full build tools"
echo "  - Install vLLM v0.6.0 (stable CPU support)"
echo "  - Take 15-25 minutes to build"
echo "  - Result in 1-5 tokens/second (CPU-only)"
echo ""
echo "Changes from original:"
echo "  ✅ Better base image (Ubuntu vs Debian slim)"
echo "  ✅ Specific vLLM version (v0.6.0)"
echo "  ✅ More robust dependency installation"
echo "  ✅ Smaller model (TinyLlama 1.1B)"
echo ""
echo "⚡ FASTER OPTION: MLX-LM is already running!"
echo "   Test it: python test_api.py"
echo "   Performance: 20-60 tokens/second"
echo ""
read -p "Continue with vLLM CPU build? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Build cancelled."
    exit 1
fi

echo "🔨 Starting vLLM CPU build..."
echo "Building with platform linux/amd64 for Docker compatibility..."
echo ""

# Build with specific platform
docker build \
    --platform linux/amd64 \
    --progress=plain \
    -t vllm-cpu:fixed \
    -f Dockerfile.vllm-cpu-fixed \
    .

echo ""
echo "✅ Build complete!"
echo ""
echo "To run:"
echo "  docker run -d -p 8000:8000 --name vllm-cpu vllm-cpu:fixed"
echo ""
echo "To test:"
echo "  sleep 30  # Wait for model to load"
echo "  python test_api.py"
echo ""
echo "⚠️  Remember: This will be 10-20x slower than MLX-LM"
echo "   MLX-LM is already running on port 8000 with better performance!"

