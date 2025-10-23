#!/bin/bash
# Check vLLM Docker build progress

echo "🔍 Checking vLLM Docker build status..."
echo ""

if docker images | grep -q "vllm-cpu.*demo"; then
    echo "✅ Build complete!"
    echo ""
    docker images | grep vllm-cpu
    echo ""
    echo "To run:"
    echo "  docker run -d -p 8000:8000 --name vllm-demo vllm-cpu:demo"
    exit 0
fi

if ps aux | grep -q "[d]ocker build.*vllm-cpu-fixed"; then
    echo "⏳ Build in progress..."
    echo ""
    echo "Last 15 lines of build log:"
    echo "---"
    tail -15 /tmp/vllm_build.log 2>/dev/null || echo "Log not available yet"
    echo "---"
    echo ""
    echo "Check full log: tail -f /tmp/vllm_build.log"
else
    echo "❌ Build not running"
    echo ""
    echo "Start build with:"
    echo "  cd deploy_llm_container_locally"
    echo "  ./build_vllm_cpu_fixed.sh"
fi

