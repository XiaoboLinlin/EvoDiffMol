#!/bin/bash
# EvoDiffMol Installation Script (pip-only, for users without conda/DNS issues)
# This script installs EvoDiffMol using pip in a virtual environment

set -e  # Exit on error

echo "========================================================================"
echo "EvoDiffMol Installation Script (pip-only)"
echo "========================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if in right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    echo "Please run this script from the EvoDiffMol root directory."
    exit 1
fi

echo -e "${GREEN}✓${NC} Found EvoDiffMol package"
echo ""

# Check Python version and prefer miniconda base python to avoid env conflicts
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python not found!${NC}"
        exit 1
    fi
fi

# Check if current python is from a conda environment (not base)
CURRENT_PYTHON_PATH=$(which $PYTHON_CMD 2>/dev/null)
if [[ "$CURRENT_PYTHON_PATH" == *"/envs/"* ]]; then
    echo -e "${YELLOW}Detected conda environment python${NC}"
    echo "Looking for miniconda base python to avoid package conflicts..."
    
    # Try to use miniconda base python instead
    if [ -f "$HOME/miniconda3/bin/python3" ]; then
        REAL_PATH=$(readlink -f "$HOME/miniconda3/bin/python3")
        if [[ "$REAL_PATH" != *"/envs/"* ]]; then
            BASE_VERSION=$($HOME/miniconda3/bin/python3 --version 2>&1 | awk '{print $2}')
            BASE_MAJOR=$(echo $BASE_VERSION | cut -d. -f1)
            BASE_MINOR=$(echo $BASE_VERSION | cut -d. -f2)
            if [ "$BASE_MAJOR" -ge 3 ] && [ "$BASE_MINOR" -ge 11 ]; then
                echo -e "${GREEN}✓${NC} Using miniconda base Python $BASE_VERSION (cleaner installation)"
                PYTHON_CMD="$HOME/miniconda3/bin/python3"
            fi
        fi
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.11
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${YELLOW}Warning: Python $PYTHON_VERSION found, but Python 3.11+ is required${NC}"
    
    # Check if conda/miniconda BASE Python (not env) is available and >= 3.11
    # Use base python to avoid conflicts with conda environment packages
    if [ -f "$HOME/miniconda3/bin/python3" ]; then
        # Check if this is the base python (not a symlink to an env)
        REAL_PATH=$(readlink -f "$HOME/miniconda3/bin/python3")
        if [[ "$REAL_PATH" != *"/envs/"* ]]; then
            CONDA_VERSION=$($HOME/miniconda3/bin/python3 --version 2>&1 | awk '{print $2}')
            CONDA_MAJOR=$(echo $CONDA_VERSION | cut -d. -f1)
            CONDA_MINOR=$(echo $CONDA_VERSION | cut -d. -f2)
            if [ "$CONDA_MAJOR" -ge 3 ] && [ "$CONDA_MINOR" -ge 11 ]; then
                echo -e "${GREEN}✓${NC} Found miniconda base Python $CONDA_VERSION"
                PYTHON_CMD="$HOME/miniconda3/bin/python3"
            fi
        fi
    fi
    
    # Final check
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        echo -e "${RED}Error: Python 3.11+ is required. Found: $PYTHON_VERSION${NC}"
        echo "Please install Python 3.11 or higher."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD ($PYTHON_VERSION)"
echo ""

# Create virtual environment
ENV_DIR="evodiff_env"
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}Warning: Virtual environment already exists${NC}"
    echo "Removing existing environment for clean installation..."
    rm -rf "$ENV_DIR"
fi

echo "Creating virtual environment..."
# Use --copies to avoid symlink issues with conda environments
$PYTHON_CMD -m venv $ENV_DIR --copies 2>/dev/null || $PYTHON_CMD -m venv $ENV_DIR
echo -e "${GREEN}✓${NC} Virtual environment created"

echo ""
echo "Activating virtual environment..."
source $ENV_DIR/bin/activate

echo -e "${GREEN}✓${NC} Environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing EvoDiffMol package and dependencies..."
echo "This may take several minutes..."
echo ""

# Install torch and torchvision first (they must be compatible versions)
echo "Installing PyTorch and torchvision..."
pip install "torch==2.5.1" "torchvision==0.20.1"

echo ""
echo "Installing EvoDiffMol and remaining dependencies (except PyG extensions)..."
pip install -e .

echo ""
echo "Installing PyTorch Geometric extensions..."
# Now detect the FINAL torch version after all other dependencies are installed
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
CUDA_VERSION=$(python -c "import torch; v=torch.version.cuda; print('cu' + v.replace('.', '') if v else 'cpu')" 2>/dev/null)

if [ -n "$TORCH_VERSION" ] && [ -n "$CUDA_VERSION" ] && [ "$CUDA_VERSION" != "cpu" ]; then
    echo "Detected: PyTorch ${TORCH_VERSION} with ${CUDA_VERSION}"
    echo "Installing from PyG wheel index for compatibility..."
    pip install --force-reinstall --no-deps torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
else
    echo "Could not detect torch/CUDA versions, installing with --no-build-isolation..."
    pip install --force-reinstall --no-deps torch-scatter torch-sparse torch-cluster --no-build-isolation
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "========================================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "To verify installation, run:"
echo '  python -c "from evodiffmol import MoleculeGenerator; print('"'"'✓ EvoDiffMol working!'"'"')"'
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""

# Run quick verification
echo "Running verification test..."
python -c "
import sys
try:
    from evodiffmol import MoleculeGenerator
    print('✓ Import successful!')
    print('✓ EvoDiffMol is ready to use!')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" && echo -e "${GREEN}✓ Verification passed!${NC}" || echo -e "${RED}✗ Verification failed!${NC}"

echo ""
echo "📥 Next Step: Download pre-trained checkpoint"
echo "  python assets/download_checkpoint.py"
echo ""

