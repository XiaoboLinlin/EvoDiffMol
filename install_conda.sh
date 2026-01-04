#!/bin/bash
# EvoDiffMol Installation Script (conda-based, recommended)

set -e  # Exit on error

echo "========================================================================"
echo "EvoDiffMol Installation Script (conda)"
echo "========================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found!${NC}"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Or use the pip-only installation:"
    echo "  bash install_pip_only.sh"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found conda"
echo ""

# Check if in right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    echo "Please run this script from the EvoDiffMol root directory."
    exit 1
fi

# Environment name
ENV_NAME="evodiff"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists${NC}"
    read -p "Remove and recreate? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment..."
        conda activate $ENV_NAME
        pip install -e .
        echo ""
        echo -e "${GREEN}✓ Package updated in existing environment!${NC}"
        exit 0
    fi
fi

echo "Creating conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo -e "${GREEN}✓${NC} Environment created"
echo ""

# Initialize conda for bash
eval "$(conda shell.bash hook)"

echo "Activating environment..."
conda activate $ENV_NAME

echo -e "${GREEN}✓${NC} Environment activated"
echo ""

echo "Installing EvoDiffMol package and dependencies..."
echo "This may take several minutes..."
echo ""

# Install package
pip install -e .

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "========================================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation, run:"
echo '  python -c "from evodiffmol import MoleculeGenerator; print('"'"'✓ EvoDiffMol working!'"'"')"'
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
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

