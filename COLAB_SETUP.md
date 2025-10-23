# Google Colab Setup Guide

## 🎯 Why Google Colab?

This course uses **Unsloth** for efficient LLM fine-tuning, which requires:
- CUDA GPUs (NVIDIA)
- Linux environment
- Significant VRAM (8-16GB)

**Google Colab provides all of this for free!**

## 🚀 Getting Started

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Sign in with your Google account

### Step 2: Upload a Lab Notebook
1. Download a notebook from `lab_notebooks/` folder
2. In Colab: **File → Upload notebook**
3. Select the lab file (e.g., `Lab3_Hello_Unsloth.ipynb`)

### Step 3: Enable GPU Runtime
**⚠️ Critical Step - Do this first!**

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Set **Hardware accelerator** to **GPU** (T4 GPU)
4. Click **Save**

### Step 4: Verify GPU Access
Run this in a code cell:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
CUDA available: True
GPU: Tesla T4
```

### Step 5: Install Dependencies
**⚠️ Updated Installation Instructions (Jan 2025)**

All Unsloth labs (4-9) now include this installation cell (**already uncommented and ready to run**):
```python
# Install Unsloth and dependencies (CRITICAL: Install in this exact order!)
# Step 1: Install Unsloth first (this is crucial!)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Step 2: Install unsloth_zoo
!pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

# Step 3: Install compatible xformers (use available version)
!pip install xformers==0.0.25.post1 --no-deps

# Step 4: Install other dependencies with version constraints
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Step 5: Install compatible transformers version
!pip install "transformers>=4.51.3,<=4.56.2" --upgrade

# Step 6: Install datasets
!pip install datasets --upgrade

print("✅ Installation complete! Now restart runtime before proceeding.")
```

**Key Improvements:**
- ✅ **Compatible xformers version**: Uses available version (0.0.25.post1)
- ✅ **Correct transformers version**: Matches unsloth-zoo requirements
- ✅ **Correct installation order**: Prevents weights/biases initialization errors
- ✅ **Import order**: Unsloth must be imported FIRST in code cells
- ⚠️ **TPU Warning**: Unsloth requires CUDA GPU, NOT TPU!

**Note**: The installation lines are already uncommented - just run the cell!

Run the cell and wait for installation (~3-5 minutes)

### Step 6: Run the Lab
- Execute cells in order (Shift + Enter)
- Follow instructions in markdown cells
- Complete TODO sections

## 📊 Colab Runtime Specs

### Free Tier
- **GPU**: Tesla T4 (16GB VRAM)
- **RAM**: 12-13GB
- **Disk**: 100GB
- **Time Limit**: ~12 hours per session
- **Restrictions**: May disconnect if idle

### Tips for Free Tier:
1. **Save often**: `File → Save` or Ctrl+S
2. **Download outputs**: Right-click files → Download
3. **Keep active**: Execute cells periodically
4. **Use checkpoints**: Save model weights regularly

### Colab Pro ($9.99/month)
- Better GPUs (V100, A100)
- Longer runtime (24 hours)
- More RAM (32GB)
- Background execution

## 🔧 Common Setup Issues

### Issue: "ImportError: cannot import name 'cached_property' from 'transformers.utils'"
**This is the most common Unsloth error!**

**Cause**: Incompatible versions between Unsloth, transformers, and xformers

**Solution**: Use the updated installation commands (see Step 5 above):
```python
# Remove old packages first (if you see this error)
!pip uninstall -y unsloth transformers xformers trl unsloth_zoo

# Install fresh with correct versions (in this exact order!)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
!pip install xformers==0.0.26.post2 --no-deps  # Pre-compiled version!
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
!pip install datasets transformers --upgrade

# Restart runtime after installation
```

Then: **Runtime → Restart runtime** and run your cells again.

### Issue: "ImportError: Unsloth: Please install unsloth_zoo"
**Solution**: The notebooks now explicitly install `unsloth_zoo` from GitHub. If you still see this error:
```python
!pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
```
Then restart runtime.

### Issue: "NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU."
**This is a TPU vs GPU issue!**

**Problem**: You're using TPU runtime, but Unsloth requires CUDA GPU.

**Solution**: 
1. **Change Runtime Type**: Runtime → Change runtime type → GPU (T4 or better)
2. **NOT TPU**: Unsloth does not work with TPU, only CUDA GPU
3. **Restart Runtime**: After changing to GPU, restart the runtime
4. **Re-run Installation**: Run the installation cell again

**Why**: Unsloth is built for CUDA GPUs and uses CUDA-specific optimizations that don't work with TPU.

### Issue: "CUDA not available"
**Solution**: 
1. Runtime → Change runtime type
2. Set Hardware accelerator to **GPU**
3. Runtime → Restart runtime

### Issue: "Out of memory"
**Solutions**:
- Reduce batch size
- Use smaller model variant
- Restart runtime to clear memory
- Consider Colab Pro

### Issue: "Module not found"
**Solution**:
```python
!pip install [package-name]
```

### Issue: "Runtime disconnected"
**Prevention**:
- Keep browser tab open
- Execute cells every ~30 minutes
- Save work frequently
- Use Colab Pro for longer sessions

## 📁 Working with Files

### Upload Files
```python
from google.colab import files
uploaded = files.upload()
```

### Download Files
```python
from google.colab import files
files.download('model.pth')
```

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

Then access files at `/content/drive/MyDrive/`

## 🎓 Lab-Specific Notes

### Lab 3: Hello Unsloth
- First time running downloads model (~4GB)
- Downloads cached for future runs
- Expect ~5-10 minutes for first run

### Labs 4-6: Optimization
- Training takes longer (10-30 minutes)
- Save checkpoints to Drive
- Can resume if disconnected

### Labs 7-8: FastAPI
- Can run on Colab but meant for local deployment
- Test API endpoints with requests library
- For production, deploy to cloud service

## ⚡ Performance Tips

### Speed Up Training
```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Use DataLoader num_workers
dataloader = DataLoader(dataset, num_workers=2)
```

### Reduce Memory Usage
```python
# Use gradient accumulation
accumulation_steps = 4

# Use smaller batch size
batch_size = 8  # instead of 32

# Clear cache periodically
torch.cuda.empty_cache()
```

### Monitor Resources
```python
# Check GPU usage
!nvidia-smi

# Check RAM usage
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

## 🔄 Typical Workflow

1. **Upload notebook** to Colab
2. **Enable GPU** runtime
3. **Install dependencies** (first cell)
4. **Mount Drive** (optional, for saving)
5. **Run lab cells** in order
6. **Save results** to Drive or download
7. **Compare with solution** notebook

## 📚 Resources

- [Colab Welcome Notebook](https://colab.research.google.com/notebooks/welcome.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro Features](https://colab.research.google.com/signup)

## 🎯 Checklist Before Starting

- [ ] Google account created
- [ ] Lab notebook downloaded
- [ ] Notebook uploaded to Colab
- [ ] GPU runtime enabled (**Critical!**)
- [ ] Dependencies installed
- [ ] GPU verified (`torch.cuda.is_available() == True`)
- [ ] Ready to learn!

## ⚠️ Important Reminders

1. **Always enable GPU** before running Unsloth labs
2. **Save frequently** - sessions can disconnect
3. **Free tier has limits** - plan accordingly
4. **Download important outputs** - they don't persist
5. **Check solutions** if stuck (`solution_notebooks/`)

---

**Ready to start?** Pick a lab from `lab_notebooks/` and begin! 🚀

