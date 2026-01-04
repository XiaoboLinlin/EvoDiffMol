# EvoDiffMol

**Molecular Generation and Optimization using Diffusion Models and Genetic Algorithms**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Checkpoint](https://img.shields.io/badge/🤗-Checkpoint-orange)](https://huggingface.co/scofieldlinlin/EvoDiffMol)

Generate and optimize drug-like molecules using diffusion models with evolutionary optimization.

---

## 🚀 Quick Start

### Installation

**Option 1: Conda (Recommended)**
```bash
git clone https://github.com/YOUR_USERNAME/EvoDiffMol.git
cd EvoDiffMol
bash install_conda.sh
conda activate evodiff
```

**Option 2: Pip (Alternative, for users without conda/DNS issues)**
```bash
git clone https://github.com/YOUR_USERNAME/EvoDiffMol.git
cd EvoDiffMol
bash install_pip_only.sh
source evodiff_env/bin/activate
```

### Download Pre-trained Checkpoint
```bash
python assets/download_checkpoint.py
```

This downloads the pre-trained model (321MB) from [Hugging Face](https://huggingface.co/scofieldlinlin/EvoDiffMol).

---

## 💡 Usage

### Basic Example
```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

# Load dataset (for metadata only)
dataset = General3D('moses', split='valid', remove_h=True)

# Initialize generator
gen = MoleculeGenerator(
    checkpoint_path="assets/checkpoints/moses_without_h_80.pt",
    model_config="assets/configs/general_without_h.yml",
    ga_config="assets/configs/moses_production.yml",
    dataset=dataset
)

# Optimize molecules for target properties
molecules = gen.optimize(
    target_properties={'qed': 0.9, 'logp': 2.5},
    population_size=100,
    generations=20
)

print(f"Generated {len(molecules)} optimized molecules!")
# molecules is a list of SMILES strings
```

### Multi-Property Optimization
```python
# Optimize for multiple properties simultaneously
molecules = gen.optimize(
    target_properties={
        'qed': 0.9,      # Drug-likeness
        'logp': 2.5,     # Lipophilicity  
        'sa': 2.0,       # Synthetic accessibility (lower is better)
        'tpsa': 60.0     # Topological polar surface area
    },
    population_size=100,
    generations=20
)
```

### ADMET Property Optimization
```python
# Optimize for ADMET properties
molecules = gen.optimize(
    target_properties={
        'qed': 0.9,
        'DILI': 0.0,           # Minimize liver toxicity
        'CYP2D6_Veith': 0.0,   # Minimize CYP2D6 inhibition
        'PPBR_AZ': 78.0        # Moderate protein binding
    },
    population_size=100,
    generations=20
)
```

---

## 📊 Supported Properties

### Basic Properties
- `qed` - Drug-likeness (0-1, higher is better)
- `logp` - Lipophilicity (-2 to 6, typical drugs: 2-3)
- `sa` - Synthetic accessibility (1-10, lower is easier)
- `tpsa` - Polar surface area (0-200, drugs: 60-140)

### ADMET Properties (40+ properties)
- **Absorption:** Caco2, HIA, Pgp inhibition
- **Distribution:** BBB, PPBR, VDss
- **Metabolism:** CYP inhibition/substrate
- **Excretion:** Clearance, half-life
- **Toxicity:** hERG, AMES, DILI

See [ADMET documentation](evodiffmol/scoring/property_configs.py) for full list.

---

## 🧪 Testing

Run the core API test:
```bash
pytest tests/test_admet_opt.py -v
```

---

## 📁 Repository Structure

```
EvoDiffMol/
├── evodiffmol/              # Core package
│   ├── generator.py         # MoleculeGenerator API
│   ├── ga/                  # Genetic algorithm
│   ├── models/              # Diffusion models
│   ├── scoring/             # Property scoring
│   └── utils/               # Utilities
├── assets/                  # Configs and checkpoint download
│   ├── configs/             # Model configs
│   ├── download_checkpoint.py
│   └── README.md
├── tests/                   # Tests
│   ├── conftest.py
│   └── test_admet_opt.py
├── install_conda.sh         # Conda installation
├── install_pip_only.sh      # Pip installation
├── pyproject.toml           # Package config
└── requirements.txt         # Dependencies
```

---

## 📖 How It Works

1. **Diffusion Model**: Generates 3D molecular structures
2. **Genetic Algorithm**: Evolves population toward target properties
3. **Property Scoring**: Evaluates molecules using RDKit and TDC ADMET predictors
4. **Optimization**: Iteratively improves population over generations

---

## 🔧 Advanced Usage

### Using Your Own Checkpoint
```python
gen = MoleculeGenerator(
    checkpoint_path="path/to/your/checkpoint.pt",
    model_config="assets/configs/general_without_h.yml",
    ga_config="assets/configs/moses_production.yml"
)
```

### Scaffold-Based Generation
```python
# Generate molecules containing a specific scaffold
molecules = gen.optimize(
    target_properties={'qed': 0.9},
    scaffold_smiles='c1ccccc1',  # Benzene ring
    population_size=100,
    generations=20
)
```
