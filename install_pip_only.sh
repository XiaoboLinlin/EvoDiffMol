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

# Check Python version
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python not found!${NC}"
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.11
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}Error: Python 3.11+ is required. Found: $PYTHON_VERSION${NC}"
    echo "Please install Python 3.11 or higher."
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($PYTHON_VERSION)"
echo ""

# Create virtual environment
ENV_DIR="evodiff_env"
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}Warning: Virtual environment already exists${NC}"
    echo "Using existing environment..."
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv $ENV_DIR
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

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

# Install the package
pip install -e .

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

