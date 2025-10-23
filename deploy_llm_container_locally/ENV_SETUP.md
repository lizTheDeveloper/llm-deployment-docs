# Environment Configuration

## Setup

Both `test_api.py` and `benchmark.py` now load credentials from a `.env` file.

### Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   # .env
   API_BASE_URL=https://your-pod-id-8000.proxy.runpod.net
   API_KEY=sk-your-api-key-here
   ```

3. **Run the scripts:**
   ```bash
   # Test API
   python3 test_api.py
   
   # Benchmark
   python3 benchmark.py 20 50
   ```

---

## Configuration Options

### For RunPod Deployment:
```bash
API_BASE_URL=https://ija84s9k1w2wg9-8000.proxy.runpod.net
API_KEY=sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86
```

### For Local MLX Server:
```bash
API_BASE_URL=http://localhost:8000
API_KEY=
```

### For Local Docker:
```bash
API_BASE_URL=http://localhost:8000
API_KEY=
```

---

## Priority Order

Configuration values are loaded in this priority (highest to lowest):

1. **Command line arguments:**
   ```bash
   python3 test_api.py --url https://example.com --api-key sk-123
   ```

2. **Environment variables:**
   ```bash
   export API_BASE_URL=https://example.com
   export API_KEY=sk-123
   python3 test_api.py
   ```

3. **.env file:**
   ```bash
   # Automatically loaded from .env
   python3 test_api.py
   ```

4. **Default fallback:**
   ```
   API_BASE_URL=http://localhost:8000
   API_KEY=None
   ```

---

## Examples

### Test RunPod (using .env):
```bash
python3 test_api.py
```

### Override with command line:
```bash
python3 test_api.py \
  --url https://different-pod-8000.proxy.runpod.net \
  --api-key sk-different-key
```

### Benchmark with custom settings:
```bash
python3 benchmark.py 50 100
```

### Use environment variables:
```bash
export API_BASE_URL=http://localhost:8000
export API_KEY=""
python3 test_api.py
```

---

## Security

⚠️ **Important:** The `.env` file contains sensitive API keys!

- ✅ `.env` is in `.gitignore` (not tracked by git)
- ✅ Never commit `.env` to version control
- ✅ Use `.env.example` for templates (without real keys)
- ✅ Share `.env.example` with your team
- ❌ Never share your actual `.env` file

---

## Troubleshooting

### "Module 'dotenv' not found"
```bash
pip install python-dotenv
```

### ".env file not loading"
Make sure `.env` is in the same directory as the scripts:
```bash
ls -la deploy_llm_container_locally/.env
```

### "401 Unauthorized"
Check your API_KEY in `.env`:
```bash
cat .env | grep API_KEY
```

### Test without .env:
```bash
python3 test_api.py \
  --url http://localhost:8000 \
  --api-key ""
```

---

## Files

```
deploy_llm_container_locally/
├── .env                # Your credentials (git-ignored)
├── .env.example        # Template file (safe to commit)
├── .gitignore          # Ignores .env and other files
├── test_api.py         # Uses .env
├── benchmark.py        # Uses .env
└── ENV_SETUP.md       # This file
```

