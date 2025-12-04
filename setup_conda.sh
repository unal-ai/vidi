#!/bin/bash
# Vidi Installation Script with Miniconda
# This script sets up the complete environment for running the Vidi project

set -e

echo "======================================"
echo "Vidi Project Installation with Conda"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER} -O /tmp/${MINICONDA_INSTALLER}
    bash /tmp/${MINICONDA_INSTALLER} -b -p $HOME/miniconda3
    rm /tmp/${MINICONDA_INSTALLER}
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    echo "Miniconda installed successfully!"
    echo "Please restart your shell or run: source ~/.bashrc"
fi

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Creating conda environment from environment.yml..."
conda env create -f environment.yml -y || conda env update -f environment.yml

echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate vidi

echo ""
echo "Step 3: Installing flash-attention (requires CUDA)..."
# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA found. Installing flash-attn..."
    pip install "flash-attn==2.6.3" --no-build-isolation || {
        echo "Warning: flash-attn installation failed. This is optional but recommended for GPU acceleration."
        echo "You may need to install it manually with: pip install flash-attn --no-build-isolation"
    }
else
    echo "Warning: CUDA not found. Skipping flash-attn installation."
    echo "Flash-attn is recommended for GPU acceleration. Install it manually if you have CUDA."
fi

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate vidi"
echo ""
echo "To run inference with Vidi-7B model:"
echo "  1. Download the model from: https://huggingface.co/bytedance-research/Vidi-7B"
echo "  2. Run: python3 Vidi_7B/inference.py --video-path <video> --query <query> --model-path <model>"
echo ""
echo "For evaluation:"
echo "  - VUE-STG: cd VUE_STG && python3 evaluate.py"
echo "  - VUE-TR-V2: cd VUE_TR_V2 && python3 qa_eval.py"
echo ""
