#!/usr/bin/env bash
# =============================================================================
# FairCLIP — Local Ubuntu Setup Script
# =============================================================================
# This script sets up the Python environment on your local Ubuntu machine.
#
# Usage:
#   chmod +x setup_local.sh
#   ./setup_local.sh
#
# What it does:
#   1. Checks for conda (installs Miniconda if missing)
#   2. Creates conda env 'fairclip' with Python 3.10
#   3. Installs PyTorch 2.2.2 with CUDA 11.8 (works on RTX 30/40 series)
#   4. Installs all other requirements
#   5. Verifies the install by importing torch and CLIP
#
# If you have CUDA 12.1, change the index-url below to cu121.
# If you have an older RTX 20 series GPU, cu118 is correct.
# =============================================================================

set -euo pipefail  # Exit on error, undefined var, or pipe failure

# --- Color output for readability -------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No color

log()  { echo -e "${GREEN}[FairCLIP setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[FairCLIP setup]${NC} $1"; }
err()  { echo -e "${RED}[FairCLIP setup ERROR]${NC} $1"; exit 1; }

# --- Step 1: Check for conda ------------------------------------------------
log "Checking for conda..."
if ! command -v conda &> /dev/null; then
    warn "Conda not found. Please install Miniconda manually:"
    warn "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    warn "  bash Miniconda3-latest-Linux-x86_64.sh"
    warn "Then re-run this script."
    err "conda missing"
fi
log "Conda found: $(conda --version)"

# --- Step 2: Create environment ---------------------------------------------
ENV_NAME="fairclip"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists."
    read -p "Remove it and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        log "Keeping existing environment. Skipping creation."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    log "Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# --- Step 3: Activate and install PyTorch -----------------------------------
# We use 'source activate' here because 'conda activate' doesn't work in bash scripts
# without sourcing conda.sh first. The two-line setup below handles both cases.
log "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

log "Installing PyTorch 2.2.2 + CUDA 11.8..."
# IMPORTANT: --index-url is critical here. Plain 'pip install torch' gets the
# CPU-only build on some systems. The cu118 URL forces the CUDA 11.8 wheel.
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Step 4: Install everything else ----------------------------------------
log "Installing remaining requirements..."
pip install -r requirements.txt

# --- Step 5: Verify install -------------------------------------------------
log "Verifying installation..."
python - <<'PYTHON_EOF'
import sys

print("Python:", sys.version)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: CUDA not available. Training will be very slow on CPU.")

import open_clip
print("OpenCLIP:", open_clip.__version__)

import clip
# Original CLIP doesn't expose __version__; we just check the import.
print("OpenAI CLIP: imported OK")

import sklearn, scipy, numpy, pandas
print("scikit-learn:", sklearn.__version__)
print("scipy:", scipy.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)

print()
print("All core imports successful.")
PYTHON_EOF

log "============================================================"
log "Setup complete. To activate the environment:"
log "    conda activate fairclip"
log "============================================================"
log "Next steps:"
log "  1. Edit configs/datasets/local_paths.yaml to point at your dataset locations"
log "  2. Run: python -m data.verify_data"
log "  3. Run: pytest tests/ -v"
log "============================================================"
