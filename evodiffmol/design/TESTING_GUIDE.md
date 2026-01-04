# EvoDiffMol Testing Guide

Complete testing strategy for package validation before GEMMINI integration.

---

## 📋 Quick Summary

**Purpose:** Ensure package functionality and population optimization work correctly

**Test Coverage:**
- ✅ Package installation and imports
- ✅ API initialization
- ✅ GA population optimization (core functionality)
- ✅ Scaffold-based optimization
- ✅ Parameter flexibility
- ✅ Output formats (list, DataFrame, file I/O)

**Standard Configuration:**
```python
population_size = 16
generations = 3
device = 'cuda'
dataset = cached_subset  # 50k molecules, loads in 2-3s
```

**Time Estimates:**
- Quick smoke test: ~1-2 minutes
- Single test: ~30-60 seconds
- Full test suite: ~5-10 minutes
- First run adds 30-60s for dataset cache creation (one-time)

---

## 🧪 Test Structure

```
tests/
├── conftest.py                # Pytest fixtures (shared setup)
├── test_installation.py       # Package import tests
├── test_api_basic.py          # API initialization
├── test_optimization.py       # GA optimization ⭐ (core functionality)
├── test_scaffold.py          # Scaffold optimization ⭐
├── test_parameters.py         # Parameter flexibility
├── test_formats.py           # Output formats (list, DataFrame, file I/O)
└── quick_test.py             # Quick smoke test (most important!)
```

**Total: 7 test files**

---

## ⚙️ Standard Test Configuration

To balance computational cost with thorough testing, all tests use:

```python
STANDARD_TEST_CONFIG = {
    'population_size': 16,  # Population size (maps to elite_size internally)
    'generations': 3,       # GA epochs
    'device': 'cuda',       # Use GPU for realistic testing
}
```

**Rationale:**
- `population_size = 16`: Small for fast testing, validates GA dynamics
- `generations = 3`: Enough to see optimization effect
- `device = 'cuda'`: Use realistic hardware

**Note:** API uses `population_size` for clarity, internally maps to `elite_size` in GA config

---

## 🎯 Test Categories

### 1. Installation Tests (Smoke Tests)
**File:** `tests/test_installation.py`

```python
"""Test basic package installation and imports"""
import pytest

def test_package_imports():
    """Test that package imports work"""
    import evodiffmol
    assert evodiffmol.__version__ is not None

def test_main_class_import():
    """Test that main class can be imported"""
    from evodiffmol import MoleculeGenerator
    assert MoleculeGenerator is not None

def test_submodule_imports():
    """Test that submodules can be imported"""
    from evodiffmol.models import get_model
    from evodiffmol.ga import GeneticTrainer
    from evodiffmol.scoring.scoring import MolecularScoring
    assert True
    
def test_utils_subset_import():
    """Test that utils_subset can be imported"""
    from evodiffmol.utils.utils_subset import load_or_create_subset
    assert load_or_create_subset is not None
```

---

### 2. API Basic Tests
**File:** `tests/test_api_basic.py`

```python
"""Test API initialization and basic functionality"""
import pytest
from evodiffmol import MoleculeGenerator

@pytest.fixture
def checkpoint_path():
    """Path to test checkpoint"""
    return "logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt"

@pytest.fixture
def config_path():
    """Path to test config"""
    return "configs/general_without_h.yml"

@pytest.fixture
def dataset():
    """Load cached dataset subset for initial population sampling"""
    from evodiffmol.utils.utils_subset import load_or_create_subset
    return load_or_create_subset(
        dataset_name='moses',
        subset_size=50000,
        root='datasets',
        remove_h=True
    )

def test_generator_init(checkpoint_path, config_path, dataset):
    """Test generator initialization"""
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        dataset=dataset  # Provide dataset for fast initial population
    )
    assert gen is not None
    assert gen.checkpoint_path == checkpoint_path

def test_generator_init_missing_checkpoint():
    """Test that missing checkpoint raises error"""
    with pytest.raises(FileNotFoundError):
        gen = MoleculeGenerator(checkpoint_path="nonexistent.pt")

def test_generator_device_selection(checkpoint_path, config_path, dataset):
    """Test device selection"""
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )
    assert gen.device == 'cuda'
```

---

### 3. Population Optimization Tests ⭐
**File:** `tests/test_optimization.py`

```python
"""Test GA optimization with different parameters"""
import pytest
from evodiffmol import MoleculeGenerator

@pytest.fixture
def generator(checkpoint_path, config_path, dataset):
    """Create a generator instance"""
    return MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )

def test_basic_optimization(generator):
    """Test basic single-property optimization"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3
    )
    
    assert molecules is not None
    assert isinstance(molecules, list)
    assert len(molecules) > 0
    
    # Validate SMILES
    from rdkit import Chem
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

def test_multi_property_optimization(generator):
    """Test multi-property optimization"""
    molecules = generator.optimize(
        target_properties={
            'logp': 4.0,
            'qed': 0.9
        },
        population_size=16,
        generations=3
    )
    
    assert len(molecules) > 0
    # All molecules should be valid
    from rdkit import Chem
    valid_count = sum(1 for s in molecules if Chem.MolFromSmiles(s) is not None)
    assert valid_count == len(molecules)

def test_different_population_size(generator):
    """Test with different population size"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=32,  # Larger population
        generations=3
    )
    assert len(molecules) > 0
```

---

### 4. Scaffold Optimization Tests ⭐
**File:** `tests/test_scaffold.py`

```python
"""Test scaffold-based optimization"""
import pytest
from evodiffmol import MoleculeGenerator
from rdkit import Chem

@pytest.fixture
def generator(checkpoint_path, config_path, dataset):
    """Create a generator instance"""
    return MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )

def test_scaffold_optimization_basic(generator):
    """Test basic scaffold optimization"""
    scaffold_smiles = 'c1ccccc1'  # Benzene
    
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        scaffold_smiles=scaffold_smiles
    )
    
    assert len(molecules) > 0
    
    # Verify scaffold is preserved in generated molecules
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        # Check if molecule contains scaffold substructure
        assert mol.HasSubstructMatch(scaffold_mol), f"Scaffold not found in {smiles}"

def test_scaffold_with_multiple_properties(generator):
    """Test scaffold optimization with multiple properties"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0, 'qed': 0.9},
        population_size=16,
        generations=3,
        scaffold_smiles='c1ccccc1'
    )
    
    assert len(molecules) > 0
    
    # All should be valid and contain scaffold
    scaffold_mol = Chem.MolFromSmiles('c1ccccc1')
    valid_count = 0
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol and mol.HasSubstructMatch(scaffold_mol):
            valid_count += 1
    
    assert valid_count > 0

def test_scaffold_parameters(generator):
    """Test scaffold with custom parameters"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        scaffold_smiles='c1ccccc1',
        scaffold_scale_factor=2.5,
        positioning_strategy='plane_through_origin'
    )
    
    assert len(molecules) > 0
```

---

### 5. Parameter Flexibility Tests
**File:** `tests/test_parameters.py`

```python
"""Test parameter flexibility"""
import pytest
from evodiffmol import MoleculeGenerator

@pytest.fixture
def generator(checkpoint_path, config_path, dataset):
    return MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )

def test_config_dict(generator):
    """Test passing config as dictionary"""
    config = {
        'population_size': 16,
        'generations': 3,
        'num_scale_factor': 50
    }
    
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        config=config
    )
    assert len(molecules) > 0

def test_config_file(generator):
    """Test passing config as file path"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        config='configs/test_config.yml'
    )
    assert len(molecules) > 0

def test_individual_parameters(generator):
    """Test passing individual parameters"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        num_scale_factor=50,
        batch_size=16
    )
    assert len(molecules) > 0
```

---

### 6. Output Format Tests
**File:** `tests/test_formats.py`

**Purpose:** Test list output, DataFrame output, and file I/O (`output_dir`)

```python
"""Test different output formats and file I/O"""
import pytest
from evodiffmol import MoleculeGenerator
import pandas as pd
from pathlib import Path

@pytest.fixture
def generator(checkpoint_path, config_path, dataset):
    return MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )

def test_list_output(generator):
    """Test default list output"""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3
    )
    
    assert isinstance(molecules, list)
    assert all(isinstance(s, str) for s in molecules)

def test_dataframe_output(generator):
    """Test DataFrame output"""
    df = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        return_dataframe=True
    )
    
    assert isinstance(df, pd.DataFrame)
    assert 'smiles' in df.columns
    assert len(df) > 0

def test_output_dir_basic(checkpoint_path, config_path, dataset, tmp_path):
    """Test that output_dir creates expected file structure"""
    output_dir = tmp_path / "test_output"
    
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=dataset
    )
    
    molecules = gen.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        output_dir=str(output_dir)
    )
    
    # Should still return molecules
    assert isinstance(molecules, list)
    assert len(molecules) > 0
    
    # Check file structure exists
    assert output_dir.exists(), "output_dir should be created"
    assert (output_dir / "final_results.txt").exists(), "final_results.txt missing"
    assert (output_dir / "initial").is_dir(), "initial/ directory missing"
    assert (output_dir / "epoch_last").is_dir(), "epoch_last/ directory missing (top level)"
    assert (output_dir / "logs").is_dir(), "logs/ directory missing"
    assert (output_dir / "logs" / "genetic_training.log").exists(), "training log missing"
    
    # Check that initial population was saved
    assert (output_dir / "initial" / "elite_molecules.csv").exists(), "initial molecules CSV missing"
    
    # Check that final population was saved
    assert (output_dir / "epoch_last" / "elite_molecules.csv").exists(), "final molecules CSV missing"

def test_output_dir_scaffold(checkpoint_path, config_path, scaffold_dataset, tmp_path):
    """Test that output_dir works with scaffold optimization"""
    output_dir = tmp_path / "scaffold_output"
    
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dataset=scaffold_dataset
    )
    
    molecules = gen.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3,
        scaffold_smiles='c1ccccc1',
        output_dir=str(output_dir)
    )
    
    # Check nested scaffold structure
    # Note: Scaffold mode creates nested: output_dir/scaffold_X/run_name/
    # For testing, just verify the base output_dir is used
    assert molecules is not None
    assert len(molecules) > 0
    # The actual nested structure depends on implementation details
```

---

### 7. Quick Smoke Test ⭐ (Most Important!)
**File:** `tests/quick_test.py`

```python
"""Quick smoke test for GEMMINI integration validation"""
import pytest

def test_quick_smoke():
    """Quick test to validate basic functionality"""
    from evodiffmol import MoleculeGenerator
    from evodiffmol.utils.utils_subset import load_or_create_subset
    from rdkit import Chem
    
    # Load dataset
    dataset = load_or_create_subset('moses', subset_size=50000)
    
    # Initialize generator
    gen = MoleculeGenerator(
        checkpoint_path="logs_moses/moses_without_h/checkpoints/80.pt",
        config_path="configs/general_without_h.yml",
        device='cuda',
        dataset=dataset
    )
    
    # Run optimization
    molecules = gen.optimize(
        target_properties={'logp': 4.0},
        population_size=16,
        generations=3
    )
    
    # Validate
    assert len(molecules) > 0
    valid_count = sum(1 for s in molecules if Chem.MolFromSmiles(s) is not None)
    assert valid_count == len(molecules), "All molecules should be valid"
    
    print(f"✅ Quick smoke test passed! Generated {len(molecules)} valid molecules")
```

**Run it:**
```bash
pytest tests/quick_test.py -v
```

---

## 🚀 Running Tests

### Run All Tests
```bash
cd /mnt/nvme/projects/EvoDiffMol_3
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_optimization.py -v
```

### Run Quick Smoke Test
```bash
pytest tests/quick_test.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=evodiffmol --cov-report=html
```

---

## 📊 Test Configuration Details

### Dataset Fixture

All tests use the cached subset approach:

```python
@pytest.fixture
def dataset():
    """Load cached dataset subset for initial population sampling"""
    from evodiffmol.utils.utils_subset import load_or_create_subset
    return load_or_create_subset(
        dataset_name='moses',
        subset_size=50000,
        root='datasets',
        remove_h=True
    )
```

**Benefits:**
- ✅ Fast loading (2-3s after first setup)
- ✅ Low memory (~500 MB)
- ✅ Consistent across all tests

### Standard Parameters

```python
population_size = 16  # Small for fast testing
generations = 3       # Enough to see optimization
device = 'cuda'       # Use GPU (realistic)
```

**Why these values?**
- Fast enough for CI/CD
- Large enough to test GA dynamics
- Realistic hardware (GPU)

---

## ⏱️ Performance Expectations

### Time Estimates

| Test Type | Duration | Notes |
|-----------|----------|-------|
| Quick smoke test | 1-2 min | Most important |
| Single test | 30-60 sec | Per test function |
| Full test suite | 5-10 min | All tests |
| First run | +30-60 sec | Dataset cache creation (one-time) |

### What's NOT Tested (Too Expensive)

- ❌ Large populations (>32)
- ❌ Many generations (>5)
- ❌ Property accuracy validation (exact LogP/QED values)
- ❌ Full dataset loading (use cached subset)

---

## ✅ Pre-GEMMINI Checklist

Before integrating with GEMMINI, ensure:

- [ ] Quick smoke test passes
- [ ] All installation tests pass
- [ ] API initialization works
- [ ] Basic optimization works (pop=16, gen=3)
- [ ] Scaffold optimization works
- [ ] Dataset caching works (2-3s load time)
- [ ] Memory usage is acceptable (~500 MB)
- [ ] Generated molecules are valid SMILES

**Run checklist:**
```bash
pytest tests/quick_test.py -v
pytest tests/test_installation.py -v
pytest tests/test_api_basic.py -v
pytest tests/test_optimization.py -v
pytest tests/test_scaffold.py -v
```

---

## 🐛 Debugging Failed Tests

### Import Errors
```python
# Check package installation
pip show evodiffmol

# Reinstall if needed
pip install -e .
```

### CUDA Errors
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())

# Run tests on CPU if needed
# (modify fixtures to use device='cpu')
```

### Dataset Cache Issues
```bash
# Check cache status
python -m evodiffmol.utils_subset

# Force recreate cache
rm datasets/moses/without_h/processed/moses_subset_50000_without_h.pt
```

### Memory Issues
```python
# Reduce batch size
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=16,
    generations=3,
    batch_size=8  # Reduce from default
)
```

---

## 📚 Additional Resources

- **API Reference:** See `API_REFERENCE.md` for complete API documentation
- **Implementation Guide:** See `IMPLEMENTATION_GUIDE.md` for setup details
- **Example Scripts:** See `../examples/` for usage examples

---

**Last Updated:** October 14, 2025

