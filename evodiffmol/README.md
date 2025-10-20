# EvoDiffMol Package

Python package for molecular generation using diffusion models and genetic algorithm optimization.

---

## 📋 Quick Reference

**👉 [PACKAGE_BUILD_SUMMARY.md](PACKAGE_BUILD_SUMMARY.md)** - Complete build summary, verification results, and quick start guide

---

## 📖 Documentation

All documentation is organized in the **`design/`** folder:

| Document | Description |
|----------|-------------|
| **[`API_REFERENCE.md`](design/API_REFERENCE.md)** | Complete API documentation for `MoleculeGenerator` |
| **[`IMPLEMENTATION_GUIDE.md`](design/IMPLEMENTATION_GUIDE.md)** | Package structure, setup, and dataset caching strategy |
| **[`TESTING_GUIDE.md`](design/TESTING_GUIDE.md)** | Testing strategy, test cases, and examples |

---

## 🚀 Quick Start

### Installation
```bash
cd /mnt/nvme/projects/EvoDiffMol_3
pip install -e .
```

### Basic Usage
```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

# 1. Load cached dataset (recommended for speed!)
dataset = load_or_create_subset(
    dataset_name='moses',
    subset_size=50000,
    root='datasets',
    remove_h=True
)
# First time: ~30-60s | Subsequent: ~2-3s ⚡

# 2. Initialize generator
gen = MoleculeGenerator(
    checkpoint_path="path/to/checkpoint.pt",
    model_config="path/to/config.yml",
    ga_config="path/to/ga_config.yml",
    device='cuda',
    dataset=dataset  # Fast initial population sampling!
)

# 3. Optimize molecules
molecules = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=100,
    generations=50
)
```

**See [`design/API_REFERENCE.md`](design/API_REFERENCE.md) for detailed usage examples.**

---

## 📁 Package Structure

```
evodiffmol/
├── __init__.py              # Package exports
├── generator.py             # MoleculeGenerator class
├── design/                  # 📖 All documentation
│   ├── API_REFERENCE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── TESTING_GUIDE.md
├── models/                  # Symlink → ../models/
├── ga/                      # Symlink → ../ga/
├── datasets/                # Symlink → ../datasets/
├── scoring/                 # Symlink → ../scoring/
└── utils/                   # Symlink → ../utils/
```

---

## 🎯 Key Features

- ✅ **Simple API:** One class (`MoleculeGenerator`), one method (`optimize()`)
- ⚡ **Fast initialization:** 2-3s with cached dataset subset
- 🎨 **Flexible configuration:** Override any parameter or use YAML configs
- 🧬 **Scaffold-based optimization:** Generate molecules with specific substructures
- 📊 **Multiple output formats:** List, DataFrame, or disk files
- 🔧 **Backward compatible:** Existing scripts continue to work

---

## 📚 Learn More

- **[API Reference](design/API_REFERENCE.md)** - Complete API documentation
- **[Implementation Guide](design/IMPLEMENTATION_GUIDE.md)** - Package setup and dataset strategy
- **[Testing Guide](design/TESTING_GUIDE.md)** - Testing strategy and examples

---

**Version:** 1.0.0  
**Target:** GEMMINI Integration  
**Status:** ✅ Complete and Verified

