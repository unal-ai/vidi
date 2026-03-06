#!/usr/bin/env bash
# setup_conda.sh — One-stop miniconda-based setup for the Vidi project.
#
# Usage:
#   bash setup_conda.sh [COMPONENT]
#
# COMPONENT can be one of:
#   vidi7b    – Set up the Vidi-7B inference environment
#   vidi9b    – Set up the Vidi1.5-9B inference/finetune environment
#   eval      – Set up the lightweight evaluation-only environment (VUE benchmarks)
#   all       – Set up all environments (default)
#
# Prerequisites:
#   - NVIDIA GPU with CUDA 12.1+ drivers
#   - Internet connection (to download miniconda and packages)
#
# The script will:
#   1. Install Miniconda (if not already installed)
#   2. Create the requested conda environment(s)
#   3. Install flash-attn (requires CUDA toolkit at build time)
#
# Example:
#   bash setup_conda.sh vidi9b
#
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
COMPONENT="${1:-all}"

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# ──────────────────────────────────────────────────────────────────────
# Step 1: Install Miniconda if missing
# ──────────────────────────────────────────────────────────────────────
install_miniconda() {
    if command -v conda &>/dev/null; then
        info "Conda already available: $(conda --version)"
        return
    fi

    if [ -x "${MINICONDA_DIR}/bin/conda" ]; then
        info "Miniconda found at ${MINICONDA_DIR}, adding to PATH."
        export PATH="${MINICONDA_DIR}/bin:$PATH"
        return
    fi

    info "Downloading Miniconda installer …"
    local installer="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "${MINICONDA_DIR}"
    rm -f "$installer"
    export PATH="${MINICONDA_DIR}/bin:$PATH"
    conda init bash 2>/dev/null || true
    info "Miniconda installed at ${MINICONDA_DIR}"
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Create conda environments
# ──────────────────────────────────────────────────────────────────────
create_env() {
    local env_name="$1"
    local env_file="$2"
    local flash_attn_version="${3:-}"

    if conda env list | grep -q "^${env_name} "; then
        warn "Environment '${env_name}' already exists. To recreate, run:"
        warn "  conda env remove -n ${env_name} && bash setup_conda.sh ${COMPONENT}"
        return
    fi

    info "Creating conda environment '${env_name}' from ${env_file} …"
    conda env create -f "${env_file}"

    # Install flash-attn inside the environment (requires CUDA build tools)
    if [ -n "$flash_attn_version" ]; then
        info "Installing flash-attn==${flash_attn_version} in '${env_name}' (this may take several minutes) …"
        conda run -n "${env_name}" pip install "flash-attn==${flash_attn_version}" --no-build-isolation || {
            warn "flash-attn installation failed. This is expected if no GPU/CUDA toolkit is available."
            warn "You can install it manually later with:"
            warn "  conda activate ${env_name} && pip install flash-attn==${flash_attn_version} --no-build-isolation"
        }
    fi

    info "Environment '${env_name}' is ready.  Activate with:"
    info "  conda activate ${env_name}"
}

setup_vidi7b() {
    create_env "vidi7b" "${SCRIPT_DIR}/Vidi_7B/environment.yml" "2.6.3"
}

setup_vidi9b() {
    create_env "vidi9b" "${SCRIPT_DIR}/Vidi1.5_9B/environment.yml" "2.8.3"
}

setup_eval() {
    create_env "vue_eval" "${SCRIPT_DIR}/VUE_TR_V2/environment.yml" ""
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
install_miniconda

case "${COMPONENT}" in
    vidi7b)  setup_vidi7b ;;
    vidi9b)  setup_vidi9b ;;
    eval)    setup_eval   ;;
    all)
        setup_vidi7b
        setup_vidi9b
        setup_eval
        ;;
    *)
        error "Unknown component '${COMPONENT}'. Use one of: vidi7b, vidi9b, eval, all"
        ;;
esac

info "Setup complete! Next steps:"
info "  1. Download model weights from HuggingFace:"
info "       Vidi-7B:    https://huggingface.co/bytedance-research/Vidi-7B"
info "       Vidi1.5-9B: https://huggingface.co/bytedance-research/Vidi1.5-9B"
info "  2. Activate your environment and run inference:"
info "       conda activate vidi7b   # or vidi9b"
info "       cd Vidi_7B              # or Vidi1.5_9B"
info "       python3 -u inference.py --video-path <video> --query <query> --model-path <model>"
