# HackMD.io Deployment Guide

**Quick, free, and GitHub-integrated documentation hosting**

HackMD.io is perfect for publishing markdown documentation directly from GitHub with zero configuration.

---

## Why HackMD.io?

- ✅ **Free for public repos** - No credit card required
- ✅ **Direct GitHub sync** - Automatically syncs with your repository
- ✅ **Zero configuration** - Just link your repo and done
- ✅ **Beautiful rendering** - Professional markdown rendering with ToC
- ✅ **Collaborative** - Real-time editing with comments
- ✅ **Book mode** - Multi-page documentation navigation

---

## Method 1: HackMD.io (Simplest - 5 minutes)

### Step 1: Sign Up

1. Go to https://hackmd.io/
2. Click "Sign Up"
3. Select "Sign in with GitHub"
4. Authorize HackMD

### Step 2: Create a New Note

1. Click "New note" or go to https://hackmd.io/new
2. In the editor, click the "⋮" menu (top right)
3. Select "Versions and GitHub Sync"

### Step 3: Connect to GitHub Repository

1. Click "Push to GitHub"
2. Select your repository: `lizTheDeveloper/llm-deployment-docs`
3. Choose the branch: `main`
4. Select the file path: `deployment_files/README.md`
5. Click "Push"

**Your first page is now live!**

### Step 4: Create a Book (Multi-Page Docs)

For the full documentation site with all guides:

1. Go to https://hackmd.io/new
2. Click "Book" mode (icon in top toolbar)
3. For each guide, create a new chapter:
   - Add `deployment_files/README.md` as home
   - Add `deployment_files/CLOUD_GPU_DEPLOYMENT_GUIDE.md`
   - Add `deployment_files/ENTERPRISE_SCALE_DEPLOYMENT.md`
   - Add `deployment_files/REAL_WORLD_DEPLOYMENT_BLOGS.md`

### Step 5: Enable GitHub Sync for Book

1. In Book mode, click Settings (gear icon)
2. Enable "GitHub Sync"
3. Select repository: `lizTheDeveloper/llm-deployment-docs`
4. Path: `deployment_files/`
5. Auto-sync: Enable

**Your entire documentation is now live and auto-updates from GitHub!**

---

## Method 2: CodiMD (Self-Hosted Alternative)

**If you want full control and self-hosting:**

### Quick Docker Setup

```bash
# On your Mac (for development) or cloud instance

# 1. Create docker-compose.yml
cat > docker-compose.yml <<'EOF'
version: "3"
services:
  database:
    image: postgres:11.6-alpine
    environment:
      - POSTGRES_USER=codimd
      - POSTGRES_PASSWORD=change_password
      - POSTGRES_DB=codimd
    volumes:
      - "database-data:/var/lib/postgresql/data"
    restart: always

  codimd:
    image: hackmdio/hackmd:latest
    environment:
      - CMD_DB_URL=postgres://codimd:change_password@database/codimd
      - CMD_USECDN=false
    depends_on:
      - database
    ports:
      - "3000:3000"
    volumes:
      - upload-data:/home/hackmd/app/public/uploads
    restart: always

volumes:
  database-data: {}
  upload-data: {}
EOF

# 2. Start CodiMD
docker-compose up -d

# 3. Access at http://localhost:3000
```

**For production:** Deploy to AWS/GCP/Azure and use same docker-compose.

---

## Method 3: GitHub Pages with Docsify (Free, Custom Domain)

**Ultra-lightweight, no build step, renders markdown directly:**

### Step 1: Create Docsify Setup

```bash
cd /Users/annhoward/src/Mastering_LLM_Deployment/deployment_files

# Create index.html
cat > index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Deployment Guide</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Production deployment guide for LLMs at enterprise scale">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'LLM Deployment Guide',
      repo: 'lizTheDeveloper/llm-deployment-docs',
      loadSidebar: true,
      subMaxLevel: 3,
      auto2top: true,
      search: {
        placeholder: 'Search...',
        noData: 'No results found',
        depth: 6
      },
      copyCode: {
        buttonText: 'Copy',
        errorText: 'Error',
        successText: 'Copied'
      }
    }
  </script>
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
  <script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-yaml.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/docsify-copy-code@2"></script>
</body>
</html>
EOF

# Create sidebar navigation
cat > _sidebar.md <<'EOF'
* [Home](README.md)

* Deployment Guides
  * [Cloud GPU Deployment](CLOUD_GPU_DEPLOYMENT_GUIDE.md)
  * [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)
  * [Real-World Case Studies](REAL_WORLD_DEPLOYMENT_BLOGS.md)

* Lab Files
  * [vLLM Dockerfile](Dockerfile.vllm)
  * [Orchestrator Dockerfile](Dockerfile.orchestrator)
  * [Docker Compose](docker-compose.yml)
  * [Tool Orchestrator](tool_orchestrator.py)

* Reference
  * [llama.cpp (Linux Only)](LLAMA_CPP_DOCKER_GUIDE_LINUX_ONLY.md)
  * [HackMD Deployment](HACKMD_DEPLOYMENT.md)
EOF
```

### Step 2: Push to GitHub

```bash
git add index.html _sidebar.md
git commit -m "Add Docsify documentation site"
git push origin main
```

### Step 3: Enable GitHub Pages

1. Go to https://github.com/lizTheDeveloper/llm-deployment-docs/settings/pages
2. Source: Deploy from a branch
3. Branch: `main`
4. Folder: `/deployment_files`
5. Click "Save"

**Your docs will be live at:**
`https://lizthedeveloper.github.io/llm-deployment-docs/`

**To add custom domain:**
1. Add file `deployment_files/CNAME` with content: `docs.yourcompany.com`
2. Add DNS CNAME record: `docs.yourcompany.com` → `lizthedeveloper.github.io`

---

## Method 4: Read the Docs (Professional, Free)

**Popular choice for open-source projects:**

### Setup

1. Go to https://readthedocs.org/
2. Sign in with GitHub
3. Click "Import a Project"
4. Select `llm-deployment-docs`
5. Name: "LLM Deployment Guide"
6. Click "Next"

### Configuration (Optional)

Create `.readthedocs.yaml` in repo root:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

mkdocs:
  configuration: deployment_files/mkdocs.yml
```

**Docs will be live at:**
`https://llm-deployment-docs.readthedocs.io/`

---

## Comparison: Choose Your Platform

| Platform | Setup Time | Cost | GitHub Sync | Custom Domain | Best For |
|----------|------------|------|-------------|---------------|----------|
| **HackMD.io** | 5 min | Free | ✅ Auto | ❌ Pro only | Quick sharing |
| **Docsify (GitHub Pages)** | 10 min | Free | ✅ Git push | ✅ Free | Full control |
| **Read the Docs** | 10 min | Free | ✅ Auto | ✅ Free | Open source projects |
| **CodiMD (self-hosted)** | 15 min | Server costs | ✅ Manual | ✅ Yes | Private/internal docs |
| **GitBook.com** | 15 min | Free tier | ✅ Auto | ✅ Free | Professional look |
| **MkDocs Material** | 10 min | Free (GH Pages) | ✅ Git push | ✅ Free | Most beautiful |

---

## Recommended: HackMD.io for Quick Start

**For your use case (getting docs live fast):**

### 3-Step Setup

```bash
# 1. Already have GitHub repo - ✅ Done

# 2. Go to HackMD.io
# - Sign in with GitHub
# - Create new note
# - Link to your repo
# - Done!

# 3. Share the link
# HackMD gives you: https://hackmd.io/@yourusername/llm-deployment
```

**Features you get:**
- ✅ Instant publishing (no build step)
- ✅ Auto-sync from GitHub commits
- ✅ Table of contents auto-generated
- ✅ Code syntax highlighting
- ✅ Search functionality
- ✅ Mobile-responsive
- ✅ Collaborative editing
- ✅ Version history

---

## Alternative: Docsify for GitHub Pages

**If you want custom domain and full control:**

### Complete Setup (10 minutes)

```bash
cd /Users/annhoward/src/Mastering_LLM_Deployment/deployment_files

# 1. Create Docsify files (already provided above)
cat > index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Deployment Guide</title>
  <meta name="description" content="Production deployment guide for LLMs at enterprise scale">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">
</head>
<body>
  <div id="app">Loading...</div>
  <script>
    window.$docsify = {
      name: 'LLM Deployment Guide',
      repo: 'lizTheDeveloper/llm-deployment-docs',
      loadSidebar: true,
      subMaxLevel: 3,
      auto2top: true,
      search: 'auto',
      copyCode: { buttonText: 'Copy', successText: 'Copied!' }
    }
  </script>
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
  <script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-yaml.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/docsify-copy-code@2"></script>
</body>
</html>
EOF

# 2. Create sidebar
cat > _sidebar.md <<'EOF'
* [Home](README.md)

* Deployment Guides
  * [Cloud GPU Deployment](CLOUD_GPU_DEPLOYMENT_GUIDE.md)
  * [Enterprise-Scale Deployment](ENTERPRISE_SCALE_DEPLOYMENT.md)
  * [Real-World Case Studies](REAL_WORLD_DEPLOYMENT_BLOGS.md)

* Reference
  * [llama.cpp (Linux Only)](LLAMA_CPP_DOCKER_GUIDE_LINUX_ONLY.md)
EOF

# 3. Push to GitHub
git add index.html _sidebar.md
git commit -m "Add Docsify documentation site"
git push origin main

# 4. Enable GitHub Pages (in browser)
# Go to: https://github.com/lizTheDeveloper/llm-deployment-docs/settings/pages
# Source: Deploy from a branch
# Branch: main, Folder: /deployment_files
# Save

# 5. Your docs are live!
# https://lizthedeveloper.github.io/llm-deployment-docs/
```

---

## Preview: What You Get

### HackMD.io
- **URL Format**: `https://hackmd.io/@username/llm-deployment`
- **Look**: Clean, focused markdown rendering
- **Features**: Real-time editing, comments, version history
- **Example**: Like Notion but for markdown

### Docsify (GitHub Pages)
- **URL Format**: `https://lizthedeveloper.github.io/llm-deployment-docs/`
- **Look**: Modern documentation theme with sidebar navigation
- **Features**: Full-text search, code copy buttons, mobile-responsive
- **Example**: Like FastAPI docs or Vue.js docs

### Read the Docs
- **URL Format**: `https://llm-deployment-docs.readthedocs.io/`
- **Look**: Professional documentation site
- **Features**: Versioning, multiple formats (HTML/PDF/ePub), search
- **Example**: Like Python official docs

---

## My Recommendation for You

**Use Docsify + GitHub Pages** because:

1. ✅ **Free forever** - No limits on public repos
2. ✅ **Custom domain** - Use docs.yourcompany.com
3. ✅ **Zero build step** - Just push markdown, it renders live
4. ✅ **Beautiful design** - Professional look out of the box
5. ✅ **Full control** - Own your content and hosting

**Setup time: 10 minutes**
**Cost: $0**
**Result: Professional documentation site**

---

## Quick Start: Docsify (Recommended)

```bash
# Run this from your Mac:

cd /Users/annhoward/src/Mastering_LLM_Deployment

# Create the files (I'll do this for you in next step)
# Then just:

git add deployment_files/index.html deployment_files/_sidebar.md
git commit -m "Add Docsify documentation site"
git push origin main

# Then enable GitHub Pages in repo settings
# Done! Your docs are live.
```

**Want me to create the Docsify files for you right now?**

---

## Summary

**Fastest (5 min):** HackMD.io - Sign in with GitHub, link repo, done

**Best for production (10 min):** Docsify + GitHub Pages - Free, custom domain, beautiful

**Most professional (10 min):** Read the Docs - Standard for open source

**For internal use (15 min):** CodiMD self-hosted - Full control, private

---

**All your markdown files work with ANY of these platforms - they're already perfect!**

Just pick your platform and follow the setup. I recommend Docsify for the best balance of simplicity and professionalism.
