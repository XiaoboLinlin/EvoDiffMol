# EvoDiffMol Implementation Guide

Complete guide for setting up and integrating EvoDiffMol as a Python package.

---

## 📋 Table of Contents

1. [Project Structure](#-project-structure)
2. [Installation](#-installation)
3. [Dataset Strategy](#-dataset-strategy)
4. [GEMMINI Integration](#-gemmini-integration)
5. [Package Development](#-package-development)

---

## 📁 Project Structure

```
EvoDiffMol_3/
├── evodiffmol/                      # Python package (for GEMMINI & pip install)
│   ├── __init__.py                  # Package initialization
│   ├── generator.py                 # Main MoleculeGenerator class
│   │
│   ├── design/                      # Design documentation
│   │   ├── API_REFERENCE.md         # API documentation
│   │   ├── IMPLEMENTATION_GUIDE.md  # This file
│   │   └── TESTING_GUIDE.md         # Testing documentation
│   │
│   ├── models/                      # Symlink → ../models/
│   ├── ga/                          # Symlink → ../ga/
│   ├── datasets/                    # Symlink → ../datasets/
│   ├── scoring/                     # Symlink → ../scoring/
│   └── utils/                       # Symlink → ../utils/
│       └── utils_subset.py          # Dataset caching utilities
│
├── models/                          # Diffusion model code
├── ga/                              # Genetic algorithm code
├── datasets/                        # Dataset utilities
├── scoring/                         # Molecular scoring functions
├── utils/                           # General utilities
├── configs/                         # Configuration files
│
├── train_genetic.py                 # Training scripts (still work!)
├── ga_train_moses_2p5k.py
├── ga_train_moses_scaffold.py
│
├── tests/                           # Test suite
│   ├── test_installation.py
│   ├── test_api_basic.py
│   ├── test_optimization.py
│   └── ...
│
└── setup.py                         # Package installer
```

**Key Points:**
- ✅ Package is created **in-place** (no code duplication)
- ✅ Uses **symlinks** to existing code directories
- ✅ **Old scripts still work** without modifications
- ✅ **New API** available for external platforms like GEMMINI

---

## 🚀 Installation

### For Development (Editable Install)

```bash
cd /mnt/nvme/projects/EvoDiffMol_3
pip install -e .
```

**Benefits:**
- Changes to code are immediately reflected
- No need to reinstall after editing
- Perfect for development and testing

### For Production

```bash
pip install git+https://github.com/your-org/EvoDiffMol.git
```

---

## 💾 Dataset Strategy

### Overview

The genetic algorithm requires an initial population to start optimization. We use **two different strategies** depending on the mode:

#### Regular Optimization (No Scaffold)
- Sample from a **cached 50k molecule subset**
- Fast loading: 2-3s after first setup
- Memory: ~500 MB
- Use case: General property optimization

#### Scaffold-Based Optimization ⚠️
- **Must filter FULL dataset** (1.9M molecules) to find scaffold matches
- First load: 10-30 minutes (scans entire dataset)
- Subsequent: 2-3s (cached filtered dataset)
- Memory: ~200-500 MB (filtered result)
- Use case: Generate molecules with specific substructure

### Quick Start

**For regular optimization (no scaffold):**
```python
from utils.utils_subset import load_or_create_subset

# Load or create cached 50k subset
dataset = load_or_create_subset(
    dataset_name='moses',
    subset_size=50000,
    root='datasets',
    remove_h=True
)
# First time: ~30-60s (creates cache)
# Subsequent times: ~2-3s (loads from cache)
```

**For scaffold-based optimization:**
```python
from utils.dataset_scaffold_smiles import create_scaffold_dataset

# Filter FULL dataset for scaffold matches
scaffold_dataset = create_scaffold_dataset(
    scaffold='c1ccccc1',  # Benzene
    dataset_name='moses',  # Loads FULL 1.9M molecules!
    max_molecules=10000
)
# First time: ~10-30 min (scans entire dataset)
# Subsequent times: ~2-3s (loads cached filtered dataset)
```

### How It Works

```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

# 1. Load cached 50k molecule subset
dataset = load_or_create_subset('moses', subset_size=50000)

# 2. Initialize generator
gen = MoleculeGenerator("checkpoint.pt", dataset=dataset)

# 3. Optimize - initial population sampled from cached subset
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=16,  # ← Sampled from 50k subset (instant!)
    generations=3
)
```

**Internal Process:**
1. **Generation 0:** Randomly sample 16 molecules from 50k cached subset (<0.1s)
2. **Generations 1-3:** Evolve using genetic algorithm
3. **Return:** Best optimized molecules

### Performance

| Metric | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| **Loading time** | 30-60s (one-time) | 2-3s |
| **Memory usage** | ~500 MB | ~500 MB |
| **Sampling speed** | <0.1s | <0.1s |
| **Cache file size** | ~500 MB on disk | Same |

### Why 50,000 Molecules?

| Aspect | Value | Reasoning |
|--------|-------|-----------|
| Typical population size | 16-100 | Your use case |
| Cached subset size | 50,000 | 500-3000x larger |
| **Diversity** | Excellent | More than enough for thousands of runs |
| **Loading speed** | 2-3s | 10-20x faster than full dataset |
| **Memory** | 500 MB | 40x less than full dataset |

### Cache Management

#### Check Cache Status
```python
from evodiffmol.utils.utils_subset import get_subset_info

info = get_subset_info('moses', subset_size=50000, remove_h=True)
if info['exists']:
    print(f"✅ Cache exists: {info['size_mb']:.1f} MB")
else:
    print("❌ Will be created on first load")
```

#### Cache Location
```
datasets/{dataset_name}/{suffix}/processed/{dataset_name}_subset_{size}_{suffix}.pt
```

Example:
```
datasets/moses/without_h/processed/moses_subset_50000_without_h.pt
```

#### Force Recreate Cache
```python
dataset = load_or_create_subset(
    'moses',
    subset_size=50000,
    force_recreate=True  # Recreate cache
)
```

#### Test Subset Loading
```bash
cd /mnt/nvme/projects/EvoDiffMol_3
python -m evodiffmol.utils_subset
```

This will test the subset loading and report timing statistics.

---

## 🌐 GEMMINI Integration

### Server Setup

```python
# In GEMMINI app.py or __init__.py

from flask import Flask, request, jsonify
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

app = Flask(__name__)

# Initialize at server startup (once)
print("🚀 Initializing EvoDiffMol generator...")

dataset = load_or_create_subset(
    dataset_name='moses',
    subset_size=50000,
    root='datasets',
    remove_h=True
)

generator = MoleculeGenerator(
    checkpoint_path='logs_moses/checkpoints/80.pt',
    model_config='configs/general_without_h.yml',      # Diffusion model architecture
    ga_config='ga_config/moses_production.yml',        # Properties (logp, qed, sa, tpsa)
    device='cuda',
    dataset=dataset
)

print("✅ Generator ready!")

@app.route('/optimize', methods=['POST'])
def optimize():
    """Handle molecule optimization requests"""
    data = request.json
    
    try:
        molecules = generator.optimize(
            target_properties=data['properties'],
            population_size=data.get('population_size', 100),
            generations=data.get('generations', 50),
            scaffold_smiles=data.get('scaffold_smiles', None)
        )
        
        return jsonify({
            'success': True,
            'molecules': molecules,
            'count': len(molecules)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'generator': 'ready'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Startup Time

- **First time (one-time setup):** ~30-60s
  - Loads full dataset
  - Creates cached subset
  - Saves to disk
  
- **Every restart:** ~2-3s
  - Loads cached subset
  - Ready to serve requests

### Example API Request

```bash
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "properties": {"logp": 4.0, "qed": 0.9},
    "population_size": 100,
    "generations": 50
  }'
```

Response:
```json
{
  "success": true,
  "molecules": ["CCO", "c1ccccc1", ...],
  "count": 100
}
```

---

## 🛠️ Package Development

### Phase 1: Symlink Structure (Current)

**Goal:** Create package for testing without moving code

**Steps:**
1. Create `evodiffmol/` directory
2. Create `__init__.py` and `generator.py`
3. Create symlinks to existing code directories
4. Create `setup.py`
5. Install with `pip install -e .`

**Status:** Ready to implement

### Phase 2: Production Structure (Future)

**Goal:** Move code into package for production

**Steps:**
1. Move code from symlinked directories into package
2. Update imports in existing scripts
3. Test backward compatibility
4. Publish to PyPI (optional)

**Status:** After Phase 1 is validated

### Creating Symlinks

```bash
cd /mnt/nvme/projects/EvoDiffMol_3/evodiffmol

# Create symlinks
ln -s ../models models
ln -s ../ga ga
ln -s ../datasets datasets
ln -s ../scoring scoring
ln -s ../utils utils
ln -s ../configs configs
```

### Verify Installation

```python
# Test imports
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset
from evodiffmol.models import get_model
from evodiffmol.ga import GeneticTrainer

print("✅ All imports successful!")
```

### Development Workflow

```bash
# 1. Make changes to code
vim models/diffusion.py

# 2. Changes are immediately available (editable install)
python test_script.py

# 3. Run tests
pytest tests/

# 4. Commit changes
git add models/diffusion.py
git commit -m "Update diffusion model"
```

---

## 📦 setup.py Template

```python
from setuptools import setup, find_packages

setup(
    name='evodiffmol',
    version='0.1.0',
    description='Molecule generation using diffusion models and genetic algorithms',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-org/EvoDiffMol',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'rdkit>=2022.09.1',
        'numpy>=1.23.0',
        'pandas>=1.5.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
```

---

## ✅ Verification Checklist

### Installation
- [ ] `pip install -e .` completes successfully
- [ ] `from evodiffmol import MoleculeGenerator` works
- [ ] Existing scripts (`train_genetic.py`) still work

### Dataset
- [ ] `load_or_create_subset()` creates cache on first run
- [ ] Cache loads in 2-3s on subsequent runs
- [ ] Memory usage is ~500 MB

### API
- [ ] `MoleculeGenerator` initializes correctly
- [ ] `optimize()` improves target properties
- [ ] Scaffold optimization preserves substructure

### GEMMINI
- [ ] Server starts in 2-3s (after first setup)
- [ ] API endpoints respond correctly
- [ ] Multiple concurrent requests work
- [ ] Error handling works properly

---

## 🎯 Next Steps

1. **Implement Phase 1:**
   - Create `__init__.py` and `generator.py`
   - Set up symlinks
   - Create `setup.py`
   - Install and test

2. **Test Integration:**
   - Run test suite
   - Test with GEMMINI
   - Verify performance

3. **Production Ready:**
   - Complete Phase 2 (code move)
   - Update documentation
   - Publish to PyPI (optional)

---

## 📚 Additional Resources

- **API Reference:** See `API_REFERENCE.md` for complete API documentation
- **Testing Guide:** See `TESTING_GUIDE.md` for testing strategies
- **Code Examples:** See `../examples/` for usage examples

---

**Last Updated:** October 14, 2025

