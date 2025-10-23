# Course Architecture Overview

## Summary

This repository has a **two-track architecture**:
1. **Course Delivery** (Google Colab + Unsloth) - For students
2. **Testing Infrastructure** (Mac M3 + Transformers) - For development

## Why This Architecture?

### The Unsloth Requirement

**Unsloth** is the core teaching library but has strict requirements:
- ✅ CUDA GPUs (NVIDIA only)
- ✅ Linux environment  
- ❌ Cannot run on Mac M3 (no CUDA)
- ❌ Cannot run on Windows natively

**Solution**: Use Google Colab (free CUDA GPUs)

## Repository Structure

```
Mastering_LLM_Deployment/
│
├── 📚 COURSE MATERIALS (Colab/Unsloth)
│   ├── lab_notebooks/           # Exercises for students
│   ├── solution_notebooks/      # Complete solutions
│   ├── COLAB_SETUP.md          # Student setup guide
│   └── README.md               # Main documentation
│
├── 🧪 TESTING INFRASTRUCTURE (Mac M3)
│   ├── python_tests/           # Mac-compatible tests
│   │   ├── venv/              # Virtual environment
│   │   ├── lab1-8_*.py        # Test implementations
│   │   ├── run_all_tests.py   # Test runner
│   │   └── README.md          # Test documentation
│   └── TESTING_SUMMARY.md     # Test results
│
└── 📋 DOCUMENTATION
    ├── QUICK_START.md          # Quick reference
    └── ARCHITECTURE.md         # This file
```

## The Two Tracks Explained

### Track 1: Course Delivery (Students)

**Platform**: Google Colab
**Library**: Unsloth
**Models**: Full-size (e.g., Llama-3-8B)
**Purpose**: Teach LLM optimization

**Student Workflow**:
1. Download lab from `lab_notebooks/`
2. Upload to Google Colab
3. Enable GPU runtime
4. Complete exercises with Unsloth
5. Check `solution_notebooks/` when stuck

**Labs using Unsloth** (Require Colab):
- Lab 3: Hello Unsloth
- Lab 4: Knowledge Distillation
- Lab 5: Model Pruning  
- Lab 6: Quantization

### Track 2: Testing Infrastructure (Instructors)

**Platform**: Mac M3
**Library**: Transformers (Unsloth alternative)
**Models**: TinyLlama-1.1B
**Purpose**: Verify concepts work

**Development Workflow**:
1. Develop concept in `python_tests/`
2. Test locally on Mac M3
3. Adapt to Unsloth for course notebooks
4. Deploy to `lab_notebooks/` and `solution_notebooks/`

**Why It's Different**:
| Aspect | Course | Testing |
|--------|--------|---------|
| Platform | Colab (CUDA) | Mac (MPS) |
| Library | Unsloth | Transformers |
| Models | 8B parameters | 1.1B parameters |
| Datasets | Full (SQuAD, IMDB) | Simplified |
| Purpose | Teaching | Concept verification |

## Lab Compatibility Matrix

| Lab | Colab | Mac M3 | Notes |
|-----|-------|--------|-------|
| Lab 1: Keras | ✅ | ✅ | Platform-independent |
| Lab 2: GradientTape | ✅ | ✅ | Platform-independent |
| Lab 3: Unsloth | ✅ | ⚠️ | Mac uses transformers |
| Lab 4: Distillation | ✅ | ⚠️ | Mac uses simplified version |
| Lab 5: Pruning | ✅ | ⚠️ | Mac uses simplified version |
| Lab 6: Quantization | ✅ | ⚠️ | Mac uses FP16 only |
| Lab 7: FastAPI | ✅ | ✅ | Platform-independent |
| Lab 8: Tool Calling | ✅ | ✅ | Platform-independent |

Legend:
- ✅ Works identically
- ⚠️ Adapted version available for testing

## Documentation Map

### For Students

1. **Start Here**: `README.md`
2. **Colab Setup**: `COLAB_SETUP.md`
3. **Quick Reference**: `QUICK_START.md`
4. **Exercises**: `lab_notebooks/`
5. **Solutions**: `solution_notebooks/`

### For Instructors

1. **Architecture**: `ARCHITECTURE.md` (this file)
2. **Testing**: `python_tests/README.md`
3. **Test Results**: `TESTING_SUMMARY.md`
4. **Run Tests**: `python run_all_tests.py`

## Key Design Decisions

### Decision 1: Unsloth as Core Library
**Why**: Industry-standard for efficient LLM fine-tuning
**Tradeoff**: Requires CUDA (Colab dependency)
**Mitigation**: Free Colab tier provides sufficient resources

### Decision 2: Dual Implementation
**Why**: Enable local development on Mac M3
**Tradeoff**: Maintain two codebases
**Benefit**: Continuous testing without Colab credits

### Decision 3: Simplified Test Versions
**Why**: Demonstrate concepts without massive compute
**Tradeoff**: Not identical to course implementations
**Benefit**: Fast iteration and CI/CD possibility

### Decision 4: TinyLlama for Testing
**Why**: Fits in Mac M3 memory, runs on MPS
**Tradeoff**: Not as capable as Llama-3-8B
**Benefit**: 10x faster testing, same API patterns

## Common Scenarios

### Scenario 1: Student Can't Run Lab Locally
**Answer**: Expected! Labs 3-6 require Colab.
**Action**: Direct to `COLAB_SETUP.md`

### Scenario 2: Instructor Wants to Test Changes
**Answer**: Use `python_tests/` for rapid iteration
**Action**: Run `python run_all_tests.py`

### Scenario 3: Adding a New Lab
**Answer**: Create both versions
**Action**: 
1. Course version (Unsloth) → `lab_notebooks/`, `solution_notebooks/`
2. Test version (Transformers) → `python_tests/`

### Scenario 4: Colab Session Expired
**Answer**: Expected with free tier
**Action**: Save work to Drive, restart runtime

## Testing Philosophy

### What We Test

✅ **Concepts work correctly**
- Distillation reduces model size
- Pruning maintains accuracy
- Quantization improves speed
- APIs respond correctly

✅ **Code runs without errors**
- All imports resolve
- Models load successfully  
- Training loops complete
- Outputs are generated

### What We Don't Test

❌ **Exact Unsloth behavior**
- Uses different library (transformers)
- Different optimization techniques
- Different model sizes

❌ **Production performance**
- Simplified datasets
- Smaller models
- Fewer training epochs

## Success Metrics

### Course Delivery
- ✅ Students can run Labs 3-6 on Colab
- ✅ All Unsloth features work correctly
- ✅ Labs teach intended concepts
- ✅ Solutions are reference-quality

### Testing Infrastructure  
- ✅ All tests pass on Mac M3
- ✅ Tests complete in < 60 seconds
- ✅ Concepts are demonstrated
- ✅ Code quality is verified

## Future Enhancements

### Potential Additions
1. **CI/CD Pipeline**: Automate test runs
2. **Docker Images**: Standardize testing environment
3. **AWS Deployment**: Examples for Labs 7-8
4. **Advanced Topics**: Multi-GPU training, quantization-aware training

### Limitations to Address
1. **INT8 Quantization**: Not supported on Mac (backend issue)
2. **Full Datasets**: Tests use subsets for speed
3. **Production Configs**: Simplified for learning

## Conclusion

This architecture enables:
- ✅ **Teaching with industry-standard tools** (Unsloth)
- ✅ **Local development capability** (Mac M3)
- ✅ **Continuous testing** (python_tests/)
- ✅ **Clear student path** (Colab → learn)

**The key insight**: Separate teaching infrastructure (Colab/Unsloth) from testing infrastructure (Mac/Transformers) while maintaining concept parity.

---

**Questions?** Check:
- `README.md` - General overview
- `COLAB_SETUP.md` - Student setup
- `python_tests/README.md` - Testing details
- `TESTING_SUMMARY.md` - What was tested

