"""
EvoDiffMol: Molecular Generation and Optimization using Diffusion Models
========================================================================

A Python package for molecule generation and property optimization using
genetic algorithms and diffusion models.

Quick Start:
-----------
```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

# Load dataset (for metadata only)
dataset = General3D('moses', split='valid', remove_h=True)

# Initialize generator
gen = MoleculeGenerator(
    checkpoint_path="path/to/checkpoint.pt",
    model_config="configs/general_without_h.yml",
    ga_config="ga_config/moses_production.yml",
    dataset=dataset
)

# Optimize molecules
molecules = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=100,
    generations=50
)
```

For detailed documentation, see: evodiffmol/design/API_REFERENCE.md
"""

import sys
import os

# Add both evodiffmol and parent directory to sys.path so absolute imports work
# - evodiffmol/: for imports like "from utils.chem import X" in package modules
# - parent/:     for imports like "from configs.datasets_config import X"
_evodiffmol_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_evodiffmol_dir)
if _evodiffmol_dir not in sys.path:
    sys.path.insert(0, _evodiffmol_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from .generator import MoleculeGenerator

__version__ = "1.0.0"
__author__ = "EvoDiffMol Team"

__all__ = [
    'MoleculeGenerator',
]

