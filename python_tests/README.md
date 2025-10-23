# Python Tests for LLM Deployment Labs

This directory contains working Python implementations of all lab notebooks, tested and verified on Mac M3.

## Environment Setup

A Python virtual environment is set up in `./venv` with all required dependencies.

### Activate the environment:
```bash
cd python_tests
source venv/bin/activate
```

### Install dependencies (if needed):
```bash
pip install tensorflow numpy matplotlib torch transformers accelerate fastapi uvicorn pydantic
```

## Lab Descriptions

### Lab 1: Keras Quick Refresher (`lab1_keras_refresher.py`)
- Basic Keras model definition, training, and evaluation
- Synthetic dataset generation
- Loss curve visualization
- **Status**: ✓ Working

### Lab 2: GradientTape Refresher (`lab2_gradient_tape.py`)
- Custom training loops with TensorFlow GradientTape
- Manual gradient descent implementation
- **Status**: ✓ Working

### Lab 3: Hello LLM/Unsloth (`lab3_hello_unsloth.py`)
- Load and run inference with Qwen2.5 (1.1B parameters)
- Measure performance metrics (tokens/sec, memory usage)
- Optimized for Mac M3 with MPS support
- **Status**: ✓ Working

### Lab 4: Knowledge Distillation (`lab4_distillation_simple.py`)
- Simplified distillation demonstration
- Teacher-student model training
- KL divergence loss for knowledge transfer
- ~91% model size reduction
- **Status**: ✓ Working

### Lab 5: Model Pruning (`lab5_pruning_simple.py`)
- L1 unstructured pruning
- Sparsity measurement
- Fine-tuning after pruning
- ~30% weight reduction demonstrated
- **Status**: ✓ Working

### Lab 6: Model Quantization (`lab6_quantization_simple.py`)
- FP16 quantization demonstration
- Model size and inference speed comparison
- ~50% size reduction (FP32 → FP16)
- **Status**: ✓ Working (Note: INT8 quantization not supported on Mac)

### Lab 7: FastAPI Deployment (`lab7_fastapi_deployment.py`)
- OpenAI-compatible API implementation
- Chat completions endpoint (`/v1/chat/completions`)
- Qwen2.5 model serving
- **Status**: ✓ Working

### Lab 8: FastAPI Tool Calling (`lab8_fastapi_tool_calling.py`)
- Tool/function calling implementation
- Weather lookup example tool
- Multi-turn conversations with tool results
- **Status**: ✓ Working

## Running the Tests

### Run all tests:
```bash
python run_all_tests.py
```

### Run individual labs:
```bash
python lab1_keras_refresher.py
python lab2_gradient_tape.py
python lab3_hello_unsloth.py
# etc...
```

## Hardware Requirements

- **Tested on**: MacBook Pro M3 with 64GB RAM
- **Minimum RAM**: 16GB (for LLM labs)
- **Storage**: ~5GB for model downloads

## Model Information

- **Qwen2.5-1.5B-Instruct**: Used for Labs 3, 7, 8
  - Size: ~2.2GB (FP16)
  - Performance: ~20 tokens/sec on M3 MPS
  - Good balance of capability and resource usage

## Notes

1. **Virtual Environment**: Always activate the venv before running tests
2. **First Run**: Labs 3, 7, 8 will download Qwen2.5 (~2GB) on first run
3. **Device Support**: Tests automatically detect and use MPS (Apple Silicon GPU) when available
4. **Simplified Labs**: Labs 4-6 use simplified implementations suitable for Mac M3 hardware
5. **Quantization**: Full INT8 quantization requires specific backend support not available on Mac

## Troubleshooting

### Out of Memory
- Close other applications
- Labs 3, 7, 8 require ~4GB RAM for model
- Consider reducing max_tokens in generation

### Slow Performance
- Ensure MPS (Apple GPU) is being used
- Check Activity Monitor for other processes
- First run may be slower due to model compilation

### Model Download Issues
- Check internet connection
- Ensure sufficient disk space (~5GB)
- Models are cached in ~/.cache/huggingface/

## Future Improvements

- Add benchmarking utilities
- Implement INT8 quantization when Mac support improves
- Add distributed inference examples
- Include model fine-tuning examples with LoRA/QLoRA

