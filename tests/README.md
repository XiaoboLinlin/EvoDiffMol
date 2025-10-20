# EvoDiffMol Test Suite

Automated tests for the EvoDiffMol package.

## Quick Start

### 1. Run Quick Smoke Test (⚡ RECOMMENDED FIRST)
```bash
pytest tests/quick_test.py -v
```
**Time:** ~1-2 minutes  
**Purpose:** Verify basic functionality before running full suite

### 2. Run Full Test Suite
```bash
pytest tests/ -v
```
**Time:** ~5-10 minutes (first run adds 30-60s for dataset caching)

### 3. Run Specific Test Categories
```bash
# Installation tests only
pytest tests/test_installation.py -v

# API tests only
pytest tests/test_api_basic.py -v

# Core optimization tests (fitness improvement)
pytest tests/test_optimization.py -v

# 3D structure verification tests
pytest tests/test_3d_structure.py -v

# Scaffold tests
pytest tests/test_scaffold.py -v
```

### 4. Skip Slow Tests
```bash
pytest tests/ -v -m "not slow"
```

## Test Configuration

All tests use a standard configuration:
- **Population size:** 16
- **Generations:** 3
- **Device:** cuda
- **Dataset:** Cached 50k subset (loads in 2-3s after first setup)

## Test Files

- `conftest.py` - Shared fixtures and configuration
- `quick_test.py` - Quick smoke test 🚀 (dataset loading, initialization)
- `test_installation.py` - Package import tests
- `test_api_basic.py` - API initialization tests
- `test_optimization.py` - Fitness improvement tests ⭐ (compares initial vs final population)
- `test_3d_structure.py` - 3D structure verification ⭐ (diffusion output + OpenBabel)
- `test_scaffold.py` - Scaffold-based optimization
- `test_formats.py` - Output format tests

## Requirements

- pytest >= 6.2.0
- All EvoDiffMol dependencies installed
- CUDA-capable GPU (for `device='cuda'` tests)
- Checkpoint file at expected location

## Expected Output

```
========================= test session starts ==========================
collected 20 items

tests/test_installation.py::test_package_imports PASSED          [  5%]
tests/test_installation.py::test_main_class_import PASSED        [ 10%]
...
tests/test_optimization.py::test_basic_optimization PASSED       [ 75%]
...

========================= 20 passed in 5.23s ===========================
```

## Troubleshooting

**Problem:** Dataset loading is slow  
**Solution:** First run creates cache (30-60s), subsequent runs are fast (2-3s)

**Problem:** Tests fail with FileNotFoundError  
**Solution:** Verify checkpoint path in `conftest.py`

**Problem:** CUDA out of memory  
**Solution:** Reduce `population_size` in standard config or use `device='cpu'`

For detailed testing documentation, see: `evodiffmol/design/TESTING_GUIDE.md`

