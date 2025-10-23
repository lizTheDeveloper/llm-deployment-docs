# Mastering LLM Deployment

A hands-on course covering LLM optimization and deployment techniques using Unsloth, FastAPI, and AWS.

## 📁 Repository Structure

```
Mastering_LLM_Deployment/
├── lab_notebooks/              # 👨‍🎓 Student exercises (with TODOs)
├── solution_notebooks/         # ✅ Complete solutions
├── python_tests/              # 🧪 Local testing (Mac M3 compatible)
├── docs/                      # 📖 Production deployment guides
│   ├── LAB_8_VLLM_DEPLOYMENT.md
│   ├── LAB_9_TOOL_CALLING.md
│   ├── CLOUD_GPU_DEPLOYMENT_GUIDE.md
│   ├── ENTERPRISE_SCALE_DEPLOYMENT.md
│   └── REAL_WORLD_DEPLOYMENT_BLOGS.md
├── plans/                     # 📋 Course planning docs
└── devlog/                    # 📝 Development logs
```

## 🔗 Quick Links

### 📊 Course Materials
- **[Course Slides & All Labs](https://docs.google.com/presentation/d/1-FTmWgVct1Ydkwvyy8ZR-mFl7KGbH-TzZRZvfMk5aRo/edit?slide=id.g39bdb786812_0_79)** - Complete course presentation

### 📚 Lab Notebooks
- [Lab 1: Keras Quick Refresher](lab_notebooks/Lab1_Keras_Quick_Refresher.ipynb)
- [Lab 2: GradientTape Refresher](lab_notebooks/Lab2_GradientTape_Refresher.ipynb)
- [Lab 3: PyTorch Fundamentals](lab_notebooks/Lab3_PyTorch_Fundamentals.ipynb)
- [Lab 4: Hello Unsloth](lab_notebooks/Lab4_Hello_Unsloth.ipynb)
- [Lab 5: Knowledge Distillation with SQuAD](lab_notebooks/Lab5_Distillation_Unsloth_SQuAD.ipynb)
- [Lab 6: Model Pruning with SST-2](lab_notebooks/Lab6_Pruning_Unsloth_SST2.ipynb)
- [Lab 7: Quantization with IMDB](lab_notebooks/Lab7_Quantization_Unsloth_IMDB.ipynb)
- [Lab 8: FastAPI OpenAI-Compatible API](lab_notebooks/Lab8_Deployment_OpenAI_Compatible_FastAPI.ipynb)
- [Lab 9: FastAPI Tool Calling with vLLM](lab_notebooks/Lab9_Deployment_OpenAI_Compatible_FastAPI_With_Tool_Calling.ipynb)

### 📖 Deployment Guides
- **[Lab 8: Production vLLM Deployment](docs/LAB_8_VLLM_DEPLOYMENT.md)** - Deploy your first LLM with Docker (30 min)
- **[Lab 9: FastAPI Tool Calling](docs/LAB_9_TOOL_CALLING.md)** - Two-tier system (FastAPI + vLLM) (45 min)
- **[Cloud GPU Deployment Guide](docs/CLOUD_GPU_DEPLOYMENT_GUIDE.md)** - AWS/GCP/Azure deployment
- **[Enterprise-Scale Deployment](docs/ENTERPRISE_SCALE_DEPLOYMENT.md)** - Kubernetes, multi-model serving, 70B+ models
- **[Real-World Deployment Case Studies](docs/REAL_WORLD_DEPLOYMENT_BLOGS.md)** - Blog posts from Salesforce, Meta, Anthropic
- **[GKE kubectl Commands Guide](docs/GKE_KUBECTL_COMMANDS.md)** - GKE Autopilot & kubectl reference
- **[GCP Test Deployment](docs/GCP_TEST_DEPLOYMENT.md)** - Small-scale GCP deployment (400 users)
- **[llama.cpp Docker Guide](docs/LLAMA_CPP_DOCKER_GUIDE_LINUX_ONLY.md)** - Linux-only alternative

### ✅ Solutions & Tests
- **[Complete Lab Solutions](SOLUTIONS.md)** - All solution notebooks (Labs 1-9)
- **[Python Tests](python_tests/)** - Mac M3 compatible test versions
- **[Deployment Documentation](docs/README.md)** - Full documentation index

## 🎯 Two-Track System

### **Track 1: Course Delivery (Google Colab - CUDA)**
**For Students Running Labs:**

- **Platform**: Google Colab with CUDA GPUs
- **Library**: Unsloth (requires CUDA)
- **Models**: Full-size models (e.g., Meta-Llama-3-8B)
- **Use**: `lab_notebooks/` → `solution_notebooks/`

**How to use:**
1. Upload notebooks to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Follow lab instructions with Unsloth

### **Track 2: Local Testing (Mac M3)**
**For Course Development/Testing:**

- **Platform**: MacBook Pro M3 with MPS
- **Library**: Transformers (Mac-compatible)
- **Models**: TinyLlama-1.1B (resource-friendly)
- **Use**: `python_tests/` folder

**How to use:**
```bash
cd python_tests
source venv/bin/activate
python run_all_tests.py
```

## 📚 Lab Overview

### Labs 1-2: Foundations (Mac Compatible)
- **Lab 1**: Keras Quick Refresher ([Lab Notebook →](lab_notebooks/Lab1_Keras_Quick_Refresher.ipynb))
- **Lab 2**: GradientTape Refresher ([Lab Notebook →](lab_notebooks/Lab2_GradientTape_Refresher.ipynb))
- **Platform**: Works everywhere (CPU/GPU/MPS)

### Labs 3-7: Unsloth Optimization (Requires Colab)
- **Lab 3**: PyTorch Fundamentals ([Lab Notebook →](lab_notebooks/Lab3_PyTorch_Fundamentals.ipynb))
- **Lab 4**: Hello Unsloth ([Lab Notebook →](lab_notebooks/Lab4_Hello_Unsloth.ipynb))
- **Lab 5**: Knowledge Distillation with SQuAD ([Lab Notebook →](lab_notebooks/Lab5_Distillation_Unsloth_SQuAD.ipynb))
- **Lab 6**: Model Pruning with SST-2 ([Lab Notebook →](lab_notebooks/Lab6_Pruning_Unsloth_SST2.ipynb))
- **Lab 7**: Quantization with IMDB ([Lab Notebook →](lab_notebooks/Lab7_Quantization_Unsloth_IMDB.ipynb))

**⚠️ Note**: These labs require **Google Colab with CUDA**. Unsloth does not work on Mac M3.

**For Mac Testing**: See `python_tests/lab3-8_*.py` for concept demonstrations using transformers library.

### Labs 8-9: Deployment (Mac Compatible)
- **Lab 8**: FastAPI OpenAI-Compatible API ([Lab Notebook →](lab_notebooks/Lab8_Deployment_OpenAI_Compatible_FastAPI.ipynb) | [Walkthrough Guide →](docs/LAB_8_VLLM_DEPLOYMENT.md))
- **Lab 9**: FastAPI Tool Calling with vLLM ([Lab Notebook →](lab_notebooks/Lab9_Deployment_OpenAI_Compatible_FastAPI_With_Tool_Calling.ipynb) | [Walkthrough Guide →](docs/LAB_9_TOOL_CALLING.md))
- **Platform**: Works on Mac M3 (tested) or cloud GPU instances

**📖 Production Deployment Guides**: See [docs/](docs/README.md) for comprehensive cloud deployment, enterprise-scale, and real-world case studies

## 🚀 Quick Start

### For Students (Google Colab)

1. **Download a lab notebook** from `lab_notebooks/`
2. **Upload to Google Colab**
3. **Enable GPU**: Runtime → Change runtime type → T4 GPU
4. **Run the cells** - Unsloth will install automatically
5. **Check solutions** in `solution_notebooks/` when needed

### For Instructors/Developers (Mac M3)

1. **Test locally** (Mac-compatible versions):
   ```bash
   cd python_tests
   source venv/bin/activate
   python run_all_tests.py
   ```

2. **View test results**: All 8 labs tested and working

## 🔧 Technical Details

### Why Two Tracks?

**Unsloth Requirements:**
- ✅ CUDA GPU (NVIDIA)
- ✅ Linux or Google Colab
- ❌ Cannot run on Mac M3 (no CUDA support)

**Our Solution:**
- **Students**: Use Google Colab (free CUDA GPUs)
- **Testing**: Use Mac-compatible alternatives for CI/development

### What's Different in Python Tests?

| Aspect | Course (Colab) | Testing (Mac) |
|--------|----------------|---------------|
| Library | Unsloth | Transformers |
| Models | Llama-3-8B | TinyLlama-1.1B |
| GPU | CUDA | MPS (Apple Silicon) |
| Lab 4-6 | Full datasets | Simplified demos |
| Purpose | Student learning | Concept verification |

## 📖 Documentation

- **`lab_notebooks/`**: Exercise versions with TODOs and documentation links
- **`solution_notebooks/`**: Complete working solutions for Colab
- **`python_tests/README.md`**: Detailed testing documentation
- **`TESTING_SUMMARY.md`**: What was tested locally
- **`QUICK_START.md`**: Quick reference guide

## 🎓 Course Flow

### Recommended Path:

1. **Start**: Read this README
2. **Practice**: Work through `lab_notebooks/` on Google Colab
3. **Verify**: Check `solution_notebooks/` for reference
4. **Deploy**: Complete Labs 7-8 (can test locally)

### For Instructors:

1. **Develop**: Use `python_tests/` to verify concepts
2. **Test**: Run `python run_all_tests.py`
3. **Deploy**: Upload final notebooks to Colab for students
4. **Monitor**: Check student progress with solution notebooks

## ⚙️ Environment Setup

### Google Colab (Students)
```python
# Automatically handled in notebooks
!pip install unsloth transformers torch accelerate datasets
```

### Mac M3 (Testing)
```bash
cd python_tests
source venv/bin/activate  # Already configured
python run_all_tests.py
```

## 🎯 Learning Objectives

By the end of this course, students will:

1. **Understand** LLM optimization techniques (distillation, pruning, quantization)
2. **Use** Unsloth for efficient model fine-tuning
3. **Deploy** LLMs with FastAPI as OpenAI-compatible APIs
4. **Implement** tool/function calling with LLMs
5. **Optimize** models for production deployment

## 🐛 Troubleshooting

### "Unsloth won't install on my Mac"
- **Expected**: Unsloth requires CUDA
- **Solution**: Use Google Colab for Labs 3-6

### "Model is too big for Colab"
- **Solution**: Use Colab Pro or reduce batch size/model size
- **Alternative**: Some labs work with smaller models

### "Tests fail locally"
- **Check**: Virtual environment is activated
- **Check**: You're in `python_tests/` directory
- **Check**: Enough RAM available (16GB+ recommended)

## 📊 Test Results

Latest test run (Mac M3, 64GB RAM):
```
✓ Lab 1: Keras Quick Refresher          PASSED
✓ Lab 2: GradientTape Refresher         PASSED
✓ Lab 3: Hello LLM/Unsloth              PASSED (using transformers)
✓ Lab 4: Knowledge Distillation         PASSED (simplified)
✓ Lab 5: Model Pruning                  PASSED (simplified)
✓ Lab 6: Model Quantization             PASSED (FP16)
✓ Lab 7: FastAPI Deployment             PASSED
✓ Lab 8: FastAPI Tool Calling           PASSED

8/8 tests passed in 57 seconds
```

## 🤝 Contributing

When adding new labs:
1. Create exercise version in `lab_notebooks/`
2. Create solution in `solution_notebooks/` (for Colab/Unsloth)
3. Create test version in `python_tests/` (Mac-compatible)
4. Update this README

## 📝 Notes

- **Unsloth is the primary teaching library** (requires Colab)
- **Python tests use alternatives** for local development only
- **Students should use Colab** for Labs 3-6
- **Labs 7-8 work on any platform** (FastAPI)

## 🔗 Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Google Colab](https://colab.research.google.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## ✅ Solutions

**[View Complete Lab Solutions →](SOLUTIONS.md)**

All solution notebooks and Python test versions are available on the dedicated solutions page.

---

**Course Design**: Optimized for Google Colab delivery with local Mac testing capability

