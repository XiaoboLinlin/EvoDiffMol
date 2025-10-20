# EvoDiffMol Examples

Complete guide to using EvoDiffMol for molecular generation and optimization.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Activate environment
conda activate evodiff

# 2. Run your first example
python examples/01_single_property.py

# 3. Check results
cat examples/output/01_qed_optimization/epoch_last/elite_molecules.csv
```

## 📚 Example Scripts

| # | Script | Level | Description | Runtime |
|---|--------|-------|-------------|---------|
| 1 | `01_single_property.py` | Beginner | Optimize for QED (drug-likeness) | ~2-3 min |
| 2 | `02_multi_property.py` | Beginner | Multi-property optimization (LogP, QED, SA) | ~2-3 min |
| 3 | `03_dataframe_output.py` | Intermediate | Get results as pandas DataFrame | ~2-3 min |
| 4 | `04_scaffold_based.py` | Intermediate | Generate molecules with specific scaffold | ~3-4 min |
| 5 | `05_property_exploration.py` | Advanced | Compare different property targets | ~6-8 min |
| 6 | `06_batch_processing.py` | Advanced | Run multiple optimization jobs | ~10-12 min |

**All examples use:** population_size=32, generations=3 (fast demos)

## 💡 Basic Usage

### Single Property Optimization

```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

# Load dataset (for metadata only)
dataset = General3D('moses', split='valid', remove_h=True)

# Initialize generator
gen = MoleculeGenerator(
    checkpoint_path="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt",
    model_config="configs/general_without_h.yml",
    ga_config="ga_config/moses_production.yml",
    dataset=dataset
)

# Optimize for QED (drug-likeness)
molecules = gen.optimize(
    target_properties={'qed': 0.95},
    population_size=32,        # For demos; use 100-200 for production
    generations=3,             # For demos; use 20-30 for production
    output_dir='results/qed_optimization'
)

print(f"Generated {len(molecules)} optimized molecules")
```

### Multi-Property Optimization

```python
# Optimize for LogP, QED, and Synthetic Accessibility
molecules = gen.optimize(
    target_properties={
        'logp': 2.5,   # Lipophilicity
        'qed': 0.9,    # Drug-likeness
        'sa': 2.0      # Synthetic accessibility (lower is better)
    },
    population_size=200,
    generations=30,
    output_dir='results/multi_property'
)
```

### DataFrame Output

```python
# Return as DataFrame with all properties
df = gen.optimize(
    target_properties={'qed': 0.95, 'logp': 3.0},
    population_size=100,
    generations=20,
    return_dataframe=True,  # Returns pandas DataFrame
    output_dir='results/dataframe_output'
)

# Analyze results
print(df.head())
print(f"Average QED: {df['qed'].mean():.3f}")
print(f"Average LogP: {df['logp'].mean():.3f}")
```

### Scaffold-Based Optimization

```python
# Generate molecules containing a benzene scaffold
molecules = gen.optimize(
    target_properties={'qed': 0.9},
    scaffold_smiles='c1ccccc1',  # Benzene scaffold
    population_size=100,
    generations=20,
    output_dir='results/scaffold_benzene'
)
```

## 🎯 Available Properties

| Property | Code | Typical Range | Description |
|----------|------|---------------|-------------|
| **QED** | `qed` | 0.0 - 1.0 | Drug-likeness (higher = more drug-like) |
| **LogP** | `logp` | -2.0 - 6.0 | Lipophilicity (2-3 typical for drugs) |
| **SA** | `sa` | 1.0 - 10.0 | Synthetic accessibility (lower = easier) |
| **TPSA** | `tpsa` | 0 - 200 | Polar surface area (60-140 for CNS drugs) |

### Good Starting Targets

```python
# Balanced drug-like molecule
{'qed': 0.9, 'logp': 2.5, 'sa': 2.0}

# High permeability
{'qed': 0.85, 'logp': 3.5, 'tpsa': 60}

# Easy to synthesize
{'qed': 0.9, 'sa': 1.5}

# CNS penetration
{'qed': 0.9, 'logp': 2.0, 'tpsa': 70}
```

## 📂 Output Structure

When you specify `output_dir`, EvoDiffMol saves:

```
results/my_optimization/
├── config/
│   └── ga_config_used.yml          # Configuration used (for reproducibility)
├── initial/
│   └── elite_molecules.csv         # Initial population
├── epoch_1/
│   ├── elite_molecules.csv         # Population after epoch 1
│   └── mol2_files/                 # 3D structures
├── epoch_last/
│   ├── elite_molecules.csv         # Final optimized molecules ⭐
│   └── mol2_files/                 # Final 3D structures
└── logs/
    └── training.log                # Training logs
```

### CSV File Format

The `elite_molecules.csv` files contain:

| Column | Description |
|--------|-------------|
| `molecule_id` | Unique ID |
| `smiles` | SMILES string |
| `fitness_score` | Overall fitness (higher is better) |
| `logp` | LogP value |
| `qed` | QED value |
| `sa` | SA score |
| `tpsa` | TPSA value |

## ⚙️ Configuration Options

### All Parameters

```python
molecules = gen.optimize(
    # Target properties (required)
    target_properties={'qed': 0.9},
    
    # Population settings
    population_size=200,          # Number of elite molecules to keep
    generations=30,               # Number of GA iterations
    
    # Generation settings
    batch_size=128,               # GPU batch size (larger = faster)
    num_scale_factor=2.0,         # Generate 2x population per iteration
    
    # Scaffold settings (optional)
    scaffold_smiles=None,         # SMILES of required scaffold
    
    # Output settings
    output_dir='results/my_run',  # Where to save results
    save_results=True,            # Save intermediate results
    verbose=True,                 # Print progress
    
    # Return format
    return_dataframe=False        # True = DataFrame, False = list of SMILES
)
```

### Speed Optimization

```python
# Faster: Larger batch size (uses GPU more efficiently)
molecules = gen.optimize(
    target_properties={'qed': 0.95},
    batch_size=128,  # Default: 32 (increase if GPU memory allows)
    generations=20
)

# Faster: Generate fewer molecules per generation
molecules = gen.optimize(
    target_properties={'qed': 0.95},
    num_scale_factor=1.5,  # Default: 2.0 (generate 1.5x vs 2x population)
    generations=20
)
```

### Quality Optimization

```python
# Better results: Larger population
molecules = gen.optimize(
    target_properties={'qed': 0.9},
    population_size=200,  # Default: 100 (more diversity)
    generations=30
)

# Better results: More iterations
molecules = gen.optimize(
    target_properties={'qed': 0.9},
    generations=50  # Default: 15 (more optimization time)
)
```

## 📖 Learning Path

### Day 1: Basics (30 minutes)
1. Run `01_single_property.py` (3 min)
2. Explore output CSV files (10 min)
3. Run `02_multi_property.py` (4 min)
4. Compare results (10 min)

### Day 2: Analysis (1 hour)
1. Run `03_dataframe_output.py`
2. Learn pandas analysis techniques
3. Visualize property distributions
4. Filter molecules by criteria

### Day 3: Advanced (2 hours)
1. Run `04_scaffold_based.py`
2. Try different scaffolds
3. Run `05_property_exploration.py`
4. Compare results across targets

### Day 4: Production
1. Run `06_batch_processing.py`
2. Set up automated workflows
3. Scale up population/generations
4. Analyze large datasets

## 🔧 Troubleshooting

### GPU Out of Memory

```python
# Solution 1: Reduce batch_size
batch_size=32  # or 16

# Solution 2: Reduce population_size
population_size=50  # or 30
```

### Slow Performance

```python
# Solution 1: Increase batch_size (if GPU allows)
batch_size=128  # or 256

# Solution 2: Reduce num_scale_factor
num_scale_factor=1.5  # Generate fewer molecules per iteration
```

### No Improvement in Results

- Increase `generations` (try 20-30 instead of 10)
- Increase `population_size` (try 150-200 for more diversity)
- Check if target properties are realistic
- Review fitness scores in CSV files to track progress

### Installation Issues

```bash
# Verify environment
conda activate evodiff
python -c "from evodiffmol import MoleculeGenerator; print('✓ Ready!')"

# If import fails, check dependencies
pip list | grep -i torch
pip list | grep -i rdkit
```

## 📊 Example Results

After running an example, you can analyze results:

```bash
# View final molecules
cat examples/output/01_qed_optimization/epoch_last/elite_molecules.csv

# Or use pandas for analysis
python -c "
import pandas as pd
df = pd.read_csv('examples/output/01_qed_optimization/epoch_last/elite_molecules.csv')
print(df.describe())
print(f'\nTop 5 molecules:')
print(df.nlargest(5, 'fitness_score')[['smiles', 'fitness_score', 'qed']])
"
```

## 🎓 Advanced Topics

### Custom Fitness Function

The default fitness uses harmonic mean of normalized property differences. To customize, modify `ga_config.yml`:

```yaml
genetic_algorithm:
  use_harmonic_mean: true  # Set to false for arithmetic mean
```

### Batch Processing Multiple Targets

See `06_batch_processing.py` for examples of:
- Running multiple optimization jobs sequentially
- Collecting and comparing results
- Generating summary reports

### Integration with Existing Workflows

```python
# Save results for further processing
df = gen.optimize(
    target_properties={'qed': 0.9},
    population_size=100,
    generations=20,
    return_dataframe=True
)

# Export in different formats
df.to_csv('results.csv', index=False)
df.to_excel('results.xlsx', index=False)
df.to_json('results.json', orient='records')

# Or get SMILES list for RDKit
smiles_list = df['smiles'].tolist()
```

## 📚 Additional Resources

- **API Reference:** [../evodiffmol/design/API_REFERENCE.md](../evodiffmol/design/API_REFERENCE.md)
- **Testing Guide:** [../evodiffmol/design/TESTING_GUIDE.md](../evodiffmol/design/TESTING_GUIDE.md)
- **Implementation Guide:** [../evodiffmol/design/IMPLEMENTATION_GUIDE.md](../evodiffmol/design/IMPLEMENTATION_GUIDE.md)
- **Run Tests:** `pytest tests/ -v`

## ❓ FAQ

**Q: How long does optimization take?**  
A: Depends on parameters. Small runs (population=50, generations=10) take ~2-3 minutes. Production runs (population=200, generations=30) take ~30-60 minutes.

**Q: Can I use my own dataset?**  
A: Yes! The dataset is only used for metadata (atom types, size distribution). The diffusion model generates new molecules, not samples from the dataset.

**Q: What GPU do I need?**  
A: Minimum: 8GB VRAM. Recommended: 16GB+ for larger batch sizes.

**Q: How do I choose target property values?**  
A: Start with typical drug-like values: `{'qed': 0.9, 'logp': 2.5, 'sa': 2.0}`. Adjust based on your specific needs.

**Q: Can I optimize for my own properties?**  
A: Currently supports QED, LogP, SA, TPSA. To add custom properties, modify `evodiffmol/scoring/scoring.py`.

---

**Ready to start?** 🎉

```bash
conda activate evodiff
python examples/01_single_property.py
```

Happy molecule generation! 🧬✨
