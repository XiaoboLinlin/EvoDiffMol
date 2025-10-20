# EvoDiffMol API Reference

Complete API documentation for the MoleculeGenerator class.

---

## 📦 Installation

```bash
cd /mnt/nvme/projects/EvoDiffMol_3
pip install -e .
```

---

## 🚀 Quick Start

```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

# Load cached dataset subset
dataset = load_or_create_subset('moses', subset_size=50000)

# Initialize generator
gen = MoleculeGenerator(
    checkpoint_path="path/to/checkpoint.pt",
    dataset=dataset
)

# Optimize molecules
molecules = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=100,
    generations=50
)
```

---

## 📌 MoleculeGenerator Class

### `__init__(checkpoint_path, model_config=None, ga_config=None, device='cuda', dataset=None)`

Initialize the molecule generator with a trained model.

**Parameters:**
- `checkpoint_path` (str): Path to model checkpoint (.pt file)
- `model_config` (str, optional): Path to **diffusion model config** (.yml file, e.g., `configs/general_without_h.yml`). Defines model architecture. If None, attempts auto-detection.
- `ga_config` (str, optional): Path to **GA config** (.yml file, e.g., `ga_config/moses_production.yml`). Defines properties and GA parameters. If None, uses defaults.
- `device` (str, optional): Device for inference ('cuda', 'cpu'). Default: 'cuda'
- `dataset` (Dataset, optional): Cached dataset subset for sampling initial population. Use `load_or_create_subset()` to create. Required for optimal performance.

**Config Files Explained:**
- **Model config** (`configs/general_without_h.yml`): Defines the diffusion model architecture (network, layers, etc.)
- **GA config** (`ga_config/moses_production.yml`): Defines available properties (logp, qed, sa, tpsa) and GA parameters

**Example:**
```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

# Load cached dataset subset (recommended)
dataset = load_or_create_subset(
    dataset_name='moses',
    subset_size=50000,
    root='datasets',
    remove_h=True
)

# Initialize with both configs
gen = MoleculeGenerator(
    checkpoint_path="logs_moses/checkpoints/80.pt",
    model_config="configs/general_without_h.yml",      # Model architecture
    ga_config="ga_config/moses_production.yml",        # Properties & GA params
    device='cuda',
    dataset=dataset  # Fast initial population sampling
)
```

**Simplified (auto-detect configs):**
```python
# If configs are in standard locations
gen = MoleculeGenerator(
    checkpoint_path="logs_moses/checkpoints/80.pt",
    dataset=dataset
)
# Will auto-detect configs in standard paths
```

**Initial Population Strategy:**
- Initial population is sampled from cached dataset subset (50k molecules)
- Fast loading: 2-3s after first setup, ~500 MB memory
- See `IMPLEMENTATION_GUIDE.md` for details

---

## 🔬 Method: `optimize()`

### `optimize(target_properties, population_size=None, generations=None, **ga_params)`

Optimize molecules using genetic algorithm.

**Parameters:**
- `target_properties` (dict): Target properties to optimize
  - Example: `{'logp': 4.0, 'qed': 0.9}`
  
- `population_size` (int, optional): Population size for GA
  - Internally maps to `elite_size` in GA config
  - Default: From config file or 100
  
- `generations` (int, optional): Number of GA generations
  - Internally maps to `ga_epochs` in GA config
  - Default: From config file or 50

- **Scaffold parameters (optional):**
  - `scaffold_smiles` (str): SMILES string of scaffold to maintain
  - `scaffold_scale_factor` (float): Scale factor for scaffold generation (default: 2.5)
  - `positioning_strategy` (str): Scaffold positioning strategy (default: 'plane_through_origin')
  - `fine_tune_epochs` (int): Fine-tuning epochs for scaffold (default: 0)

- **Advanced GA parameters:**
  - `num_scale_factor` (int): Molecules generated per generation (default: 100)
  - `batch_size` (int): Batch size for generation (default: 16)
  - `gpu_id` (int): GPU device ID (default: 0)
  - `config` (dict or str): Full GA config dict or path to YAML file

- **Output control:**
  - `output_dir` (str, optional): Directory to save detailed results. If None (default), results are returned only (no disk writes)
  - `save_results` (bool): Save final results and logs (default: False)
  - `verbose` (bool): Print progress information (default: True)

**Returns:**
- `list` or `DataFrame`: **Final elite population** after optimization
  - These are the best molecules after all GA generations
  - Default: List of SMILES strings (e.g., `['CCO', 'c1ccccc1', ...]`)
  - With `return_dataframe=True`: Pandas DataFrame with SMILES and calculated properties
  
**What you get:**
- The **final optimized population** (elite molecules after evolution)
- Not just the best molecule, but the entire final elite set
- Size typically equals `population_size` parameter
- Sorted by fitness (best molecules first, if applicable)

**Output Files (when `output_dir` is specified):**
```
output_dir/                        # Your specified directory
├── final_results.txt              # Summary statistics
├── final_elite_population.pt      # Final population (PyTorch format)
│
├── initial/                       # Initial population (epoch 0)
│   └── elite_molecules.csv        # Initial molecules
│
├── epoch_last/                    # Final epoch results
│   ├── elite_molecules.csv        # Final population with properties
│   └── mol2_files/                # 3D structures (optional)
│       └── elite_*.mol2
│
└── logs/                          # Training logs
    └── genetic_training.log       # Detailed progress logs
```

**Note:** This matches the structure used by existing training scripts for consistency.

**Examples:**

#### Basic optimization (API mode - no disk writes)
```python
# Returns final elite population directly, no files saved
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=100,
    generations=50
)
# Returns: ['CCO...', 'c1ccccc1...', ...]  ← Final 100 optimized molecules
# Perfect for GEMMINI - clean, no disk clutter
```

#### With detailed logging (research mode)
```python
# Save detailed results to disk
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=100,
    generations=50,
    output_dir='results/experiment_001',
    save_results=True
)
# Creates structured output:
# results/experiment_001/
#   ├── final_results.txt              # Summary
#   ├── final_elite_population.pt      # Population checkpoint
#   ├── initial/
#   │   └── elite_molecules.csv        # Initial population
#   ├── epoch_last/
#   │   ├── elite_molecules.csv        # Final molecules with properties
#   │   └── mol2_files/                # 3D structures (optional)
#   └── logs/
#       └── genetic_training.log       # Progress logs
```

#### Multiple properties
```python
molecules = gen.optimize(
    target_properties={
        'logp': 4.0,
        'qed': 0.9,
        'sa_score': 3.0
    },
    population_size=100,
    generations=50
)
```

#### Scaffold-based optimization
```python
# Maintain benzene ring scaffold
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=50,
    generations=30,
    scaffold_smiles='c1ccccc1',  # Benzene
    scaffold_scale_factor=2.5,
    positioning_strategy='plane_through_origin'
)
# All generated molecules will contain the benzene ring
```

#### With DataFrame output
```python
df = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=100,
    generations=50,
    return_dataframe=True
)
print(df.head())
#        smiles  logp   qed
# 0  CCO...      4.1  0.89
# 1  c1c...      3.9  0.91
```

#### Using config file
```python
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    config='path/to/ga_config.yml'
)
```

#### Advanced control
```python
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=200,        # Evolving population
    generations=100,            # Number of GA iterations
    num_scale_factor=150,       # Generate 150 molecules per generation
    batch_size=32,              # Larger batches for faster generation
    gpu_id=0                    # Use GPU 0
)
```

---

## 📊 Target Properties

### Supported Properties

Properties are defined in the GA config file (e.g., `ga_config/moses_production.yml`).

**Default supported properties:**

| Property | Description | Range | Preferred Range/Value | Direction |
|----------|-------------|-------|----------------------|-----------|
| **`logp`** | Lipophilicity (octanol-water partition coefficient) | [-4.5, 6] | [1, 3] | Higher for drug-likeness |
| **`qed`** | Quantitative Estimate of Drug-likeness | [0, 1] | Higher | Higher is better ✅ |
| **`sa`** | Synthetic Accessibility Score | [1, 10] | Lower | Lower is better (easier to synthesize) |
| **`tpsa`** | Topological Polar Surface Area (Ų) | [0, 188] | ~75 | Around 75 for good permeability |

**Property Details:**

#### `logp` - Lipophilicity
```python
# Target a specific LogP value
{'logp': 2.5}  # Preferred range: 1-3 for drug-likeness

# Common targets:
# - logp: 1-3   → Good drug-likeness
# - logp: 4-5   → High lipophilicity
# - logp: -2-0  → Hydrophilic
```

#### `qed` - Drug-likeness
```python
# Target high QED (closer to 1 is better)
{'qed': 0.9}  # Range: 0-1, higher is better

# Common targets:
# - qed: 0.9-1.0 → Excellent drug-likeness
# - qed: 0.7-0.9 → Good drug-likeness
# - qed: 0.5-0.7 → Moderate drug-likeness
```

#### `sa` - Synthetic Accessibility
```python
# Target easy synthesis (lower is better)
{'sa': 3.0}  # Range: 1-10, lower = easier to synthesize

# Common targets:
# - sa: 1-3  → Easy to synthesize ✅
# - sa: 3-6  → Moderate difficulty
# - sa: 6-10 → Difficult to synthesize
```

#### `tpsa` - Polar Surface Area
```python
# Target good permeability
{'tpsa': 75}  # Range: 0-188, ~75 is optimal

# Common targets:
# - tpsa: 60-90  → Good oral bioavailability
# - tpsa: <140   → Can cross blood-brain barrier
# - tpsa: >140   → Poor membrane permeability
```

---

### Property Specification

#### Single Property
```python
molecules = gen.optimize(
    target_properties={'logp': 2.5}
)
```

#### Multiple Properties (Multi-objective optimization)
```python
# Optimize all properties simultaneously
molecules = gen.optimize(
    target_properties={
        'logp': 2.5,    # Target LogP = 2.5
        'qed': 0.9,     # Target QED = 0.9
        'sa': 3.0,      # Target SA = 3.0
        'tpsa': 75      # Target TPSA = 75
    }
)
```

#### Query Available Properties
```python
# See what properties are available in your config
from evodiffmol import MoleculeGenerator

gen = MoleculeGenerator("checkpoint.pt", config_path="ga_config/moses_production.yml")

# Get available properties
available_props = gen.get_available_properties()
print(available_props)
# Output: ['logp', 'qed', 'sa', 'tpsa']

# Get property details
prop_info = gen.get_property_info('logp')
print(prop_info)
# Output: {
#   'range': [-4.5, 6],
#   'preferred_range': [1, 3],
#   'description': 'Lipophilicity'
# }
```

---

### Custom Property Configuration

You can customize properties by providing a custom config file:

**Create custom config:** `my_config.yml`
```yaml
scoring_operator:
  scoring_names:
    - logp
    - qed
    - molecular_weight
  
  property_config:
    logp:
      range: [-4.5, 6]
      preferred_range: [2, 4]
    qed:
      range: [0, 1]
      higher_is_better: true
    molecular_weight:
      range: [150, 500]
      preferred_value: 350
```

**Use custom config:**
```python
gen = MoleculeGenerator(
    checkpoint_path="checkpoint.pt",
    config_path="my_config.yml"  # Your custom property definitions
)

molecules = gen.optimize(
    target_properties={
        'logp': 3.0,
        'molecular_weight': 350
    }
)
```

---

## 🏗️ Scaffold-Based Optimization

Scaffold-based optimization **requires loading the FULL MOSES training dataset** (1.9M molecules) and filtering it.

### How Scaffold Mode Works

1. **Load & Filter**: Scan FULL 1.9M molecule dataset to find scaffold matches (10-30 min, one-time)
2. **Cache**: Save filtered dataset (~200-10k molecules) to disk for fast subsequent loads (2-3s)
3. **Initial Population**: Sample from filtered dataset (all molecules contain scaffold)
4. **Generation**: Model generates new molecules incorporating scaffold
5. **Optimization**: GA optimizes while maintaining scaffold

**Key Point:** You MUST use `GeneralScaffoldDataset` or `create_scaffold_dataset()` which:
- Loads the **entire MOSES training dataset** (not the 50k subset!)
- Filters to keep only scaffold-containing molecules
- Caches result as `.pt` file for instant subsequent loads

### Creating Scaffold Dataset (Required)

**Step 1: Create filtered dataset (one-time, 10-30 minutes)**

```python
from utils.dataset_scaffold import GeneralScaffoldDataset

# Option A: Using MOL2 file
scaffold_dataset = GeneralScaffoldDataset(
    scaffold_mol2_path='datasets/scaffold_examples/benzene.mol2',
    dataset_name='moses',  # Loads FULL 1.9M molecule dataset!
    split='train',
    min_molecules=200,      # Minimum required
    max_molecules=10000,    # Stop after finding this many
    remove_h=True
)

# Option B: Using SMILES directly (simpler!) ⭐ RECOMMENDED
from utils.dataset_scaffold_smiles import create_scaffold_dataset

scaffold_dataset = create_scaffold_dataset(
    scaffold='c1ccccc1',  # Just provide SMILES!
    dataset_name='moses',  # Loads FULL 1.9M molecule dataset!
    max_molecules=10000
)
# Behind the scenes:
# 1. Converts SMILES → RDKit molecule
# 2. Generates 3D coordinates
# 3. Creates temporary MOL2 file
# 4. Loads FULL MOSES training dataset (1.9M molecules)
# 5. Filters to find molecules containing scaffold
# 6. Caches filtered result (~200-10k molecules) as .pt file

# First time: 10-30 minutes (scans entire 1.9M dataset)
# Subsequent: 2-3 seconds (loads cached ~10k filtered dataset)
```

**Recommended:** Use Option B (SMILES) for simplicity! ⭐

**Step 2: Use for optimization (fast after caching)**

```python
# Initialize with scaffold dataset
gen = MoleculeGenerator(
    checkpoint_path="checkpoint.pt",
    model_config="configs/general_without_h.yml",
    ga_config="ga_config/moses_production.yml",
    dataset=scaffold_dataset  # ← Pre-filtered dataset!
)

# Optimize with scaffold constraint
molecules = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=50,
    generations=30,
    scaffold_smiles='c1ccccc1',  # Must match dataset scaffold!
    scaffold_scale_factor=2.5
)
# Initial population: sampled from scaffold_dataset (all contain benzene)
# Generated molecules: also contain benzene
# Final molecules: optimized with benzene preserved
```

---

### Performance Trade-offs

| Aspect | Regular Dataset | Scaffold Dataset |
|--------|----------------|------------------|
| **First-time setup** | 30-60s | 10-30 minutes |
| **Subsequent loads** | 2-3s | 2-3s (cached) |
| **Memory** | ~500 MB | ~200-500 MB |
| **Dataset size** | 50k (any molecules) | 200-10k (scaffold only) |
| **Use case** | General optimization | Scaffold-specific optimization |
| **Required for scaffold mode** | ❌ No | ✅ **Yes** |

### Complete Scaffold Example

```python
from evodiffmol import MoleculeGenerator
from utils.dataset_scaffold_smiles import create_scaffold_dataset

# 1. Create scaffold dataset - filters FULL MOSES dataset!
print("⏳ Filtering 1.9M molecules for benzene scaffold...")
scaffold_dataset = create_scaffold_dataset(
    scaffold='c1ccccc1',  # Benzene SMILES
    dataset_name='moses',  # Loads FULL training set!
    max_molecules=10000
)
print("✅ Found and cached scaffold dataset!")
# First time: 10-30 min (scans entire 1.9M molecule dataset)
# Subsequent: 2-3s (loads from cached filtered dataset)

# 2. Initialize generator with scaffold dataset
gen = MoleculeGenerator(
    checkpoint_path="checkpoint.pt",
    model_config="configs/general_without_h.yml",
    ga_config="ga_config/moses_production.yml",
    device='cuda',
    dataset=scaffold_dataset  # ← Must use scaffold-filtered dataset!
)

# 3. Optimize with scaffold
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=50,
    generations=30,
    scaffold_smiles='c1ccccc1',  # Must match the dataset scaffold!
    output_dir='ga_output/benzene_scaffold'  # Optional: save results
)
# All molecules (initial + generated + final) contain benzene ✅
```

**Important:**
- The `scaffold_smiles` in `optimize()` must match the scaffold used in `create_scaffold_dataset()`
- Initial population is sampled from the filtered dataset (all contain scaffold)
- Generated molecules will also contain the scaffold
- Total time: ~10-30 min (first time) or ~2-3s (subsequent runs) + optimization time

**Alternative:** If you have a MOL2 file, you can still use it directly:
```python
from utils.dataset_scaffold import GeneralScaffoldDataset

scaffold_dataset = GeneralScaffoldDataset(
    scaffold_mol2_path='datasets/scaffold_examples/benzene.mol2',
    dataset_name='moses',
    max_molecules=10000
)
```

### Parameters

- **`scaffold_smiles`** (str): SMILES string of the scaffold to incorporate
  - Example: `'c1ccccc1'` (benzene), `'c1ccncc1'` (pyridine)
  - The model will **generate** molecules containing this substructure
  
- **`scaffold_scale_factor`** (float): Controls generation scale around scaffold
  - Higher values = more atoms added around scaffold
  - Default: 2.5
  - Range: 1.0 to 5.0
  
- **`positioning_strategy`** (str): How to position scaffold in 3D space during generation
  - `'plane_through_origin'`: Position scaffold on XY plane (default)
  - `'center_of_mass'`: Center at origin by mass
  - `'geometric_center'`: Center at geometric center
  
- **`fine_tune_epochs`** (int): Additional fine-tuning iterations on scaffold
  - Default: 0
  - Use 5-10 for better scaffold preservation

- **`output_dir`** (str): Base directory for scaffold results (optional)
  - Creates nested structure: `output_dir/scaffold_<id>/<run_name>/`
  - Each run has `initial/`, `epoch_last/`, `logs/`, `config/`
  - Example: `'ga_output/moses_scaffolds'` → `ga_output/moses_scaffolds/scaffold_1/shared_initial/`
  - Allows organizing multiple scaffolds and runs hierarchically

### Example with All Parameters

```python
molecules = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=50,
    generations=30,
    scaffold_smiles='c1cc(C)ccc1',  # Toluene
    scaffold_scale_factor=3.0,
    positioning_strategy='plane_through_origin',
    fine_tune_epochs=5,
    output_dir='ga_output/toluene_derivatives'  # Save all results
)
```

**Output structure with scaffold (nested hierarchy):**
```
ga_output/moses_scaffolds/              # Base output_dir
└── scaffold_<id>/                       # Auto-generated scaffold subdirectory
    └── <run_name>/                      # Run-specific subdirectory
        ├── initial/
        │   ├── elite_molecules.csv
        │   ├── initial_population.pt   # All from scaffold dataset
        │   └── mol2_files/
        ├── epoch_last/
        │   ├── elite_molecules.csv
        │   └── mol2_files/              # Final molecules (all have scaffold)
        ├── logs/
        │   └── genetic_training.log
        ├── config/
        ├── final_elite_population.pt
        └── final_results.txt

# Example: output_dir='ga_output/moses_scaffolds' creates:
# ga_output/moses_scaffolds/scaffold_1/shared_initial/...
# ga_output/moses_scaffolds/scaffold_1/double_logp4_sa35/...
```

---

## 🔧 Advanced Usage

### Custom GA Configuration

```python
# Option 1: Pass parameters directly
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=200,
    generations=100,
    num_scale_factor=150,
    batch_size=32
)

# Option 2: Use config dict
ga_config = {
    'population_size': 200,
    'generations': 100,
    'num_scale_factor': 150,
    'batch_size': 32,
    'gpu_id': 0
}
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    config=ga_config
)

# Option 3: Use YAML config file
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    config='configs/custom_ga_config.yml'
)
```

### Batch Processing

```python
# Process multiple optimization tasks
targets = [
    {'logp': 2.0},
    {'logp': 4.0},
    {'logp': 6.0}
]

results = []
for target in targets:
    molecules = gen.optimize(
        target_properties=target,
        population_size=50,
        generations=20
    )
    results.append(molecules)
```

### Integration with GEMMINI

```python
from flask import Flask, request, jsonify
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.utils_subset import load_or_create_subset

app = Flask(__name__)

# Initialize at startup
dataset = load_or_create_subset('moses', subset_size=50000)
generator = MoleculeGenerator("checkpoint.pt", dataset=dataset)

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    
    # No disk writes by default - perfect for API!
    molecules = generator.optimize(
        target_properties=data['properties'],
        population_size=data.get('population_size', 100),
        generations=data.get('generations', 50),
        verbose=False  # Suppress progress prints for API
    )
    # Returns molecules directly, no files created
    
    return jsonify({'molecules': molecules})

@app.route('/optimize_detailed', methods=['POST'])
def optimize_detailed():
    """Optional endpoint with detailed logging"""
    data = request.json
    run_id = data.get('run_id', 'default')
    
    # Save detailed results for this specific run
    molecules = generator.optimize(
        target_properties=data['properties'],
        population_size=data.get('population_size', 100),
        generations=data.get('generations', 50),
        output_dir=f'ga_output/gemmini_runs/{run_id}',
        save_results=True
    )
    
    return jsonify({
        'molecules': molecules,
        'results_dir': f'ga_output/gemmini_runs/{run_id}'
    })
```

---

## 📝 Return Formats

### List of SMILES (default)

```python
molecules = gen.optimize(
    target_properties={'logp': 4.0},
    population_size=50
)
# Returns: Final elite population as list of SMILES
# ['CCO', 'c1ccccc1', 'CC(C)O', ...]  ← 50 optimized molecules
# These are the best molecules after GA evolution
```

### DataFrame (with properties)

```python
df = gen.optimize(
    target_properties={'logp': 4.0, 'qed': 0.9},
    population_size=50,
    return_dataframe=True
)
# Returns: Final elite population as DataFrame
#        smiles      logp   qed
# 0  CCO...          4.1  0.89  ← Best molecule
# 1  c1c...          3.9  0.91
# ...
# 49 CC(C)O...       3.8  0.87  ← 50th molecule
# 
# Total: 50 rows (final elite population)
```

---

## ⚠️ Error Handling

```python
try:
    molecules = gen.optimize(
        target_properties={'logp': 4.0},
        population_size=100,
        generations=50
    )
except FileNotFoundError:
    print("Checkpoint or config file not found")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
```

---

## 💡 Best Practices

1. **Always use cached dataset subset**
   ```python
   dataset = load_or_create_subset('moses', subset_size=50000)
   gen = MoleculeGenerator(checkpoint, dataset=dataset)
   ```

2. **For GEMMINI/API: Don't save to disk**
   ```python
   # Clean API mode - no file creation
   molecules = gen.optimize(
       target_properties={'logp': 4.0},
       population_size=100,
       generations=50,
       verbose=False  # Suppress prints
   )
   # No ga_output/ folders created!
   ```

3. **For research: Save detailed results**
   ```python
   # Research mode - save everything
   molecules = gen.optimize(
       target_properties={'logp': 4.0},
       population_size=100,
       generations=50,
       output_dir='ga_output/experiment_001',
       save_results=True
   )
   # Creates detailed logs and results
   ```

4. **Start with small populations for testing**
   ```python
   # Test first
   molecules = gen.optimize({'logp': 4.0}, population_size=16, generations=3)
   
   # Then scale up
   molecules = gen.optimize({'logp': 4.0}, population_size=200, generations=100)
   ```

5. **Use scaffold for targeted design**
   ```python
   # When you need specific substructures
   molecules = gen.optimize(
       target_properties={'logp': 4.0},
       scaffold_smiles='c1ccccc1'
   )
   ```

6. **Monitor GPU memory**
   ```python
   # For large populations, adjust batch size
   molecules = gen.optimize(
       target_properties={'logp': 4.0},
       population_size=500,
       batch_size=16  # Reduce if OOM
   )
   ```

---

## 📚 Additional Resources

- **Implementation Guide:** See `IMPLEMENTATION_GUIDE.md` for setup and dataset strategy
- **Testing Guide:** See `TESTING_GUIDE.md` for testing examples
- **Example Scripts:** See `../examples/` for complete examples

---

---

## 🔍 Property Configuration Reference

### Where Properties Are Defined

Properties are configured in the GA config YAML file:

**File:** `ga_config/moses_production.yml`
```yaml
scoring_operator:
  scoring_names:
    - logp      # Lipophilicity
    - qed       # Drug-likeness
    - sa        # Synthetic accessibility
    - tpsa      # Polar surface area
  
  property_config:
    logp:
      range: [-4.5, 6]
      preferred_range: [1, 3]
    qed:
      range: [0, 1]
      higher_is_better: true
    sa:
      range: [1, 10]
      higher_is_better: false  # Lower is better
    tpsa:
      range: [0, 188]
      preferred_value: 75
```

### How the GA Uses These Configs

1. **Range**: Valid range for the property
2. **Preferred Range/Value**: Optimal range for drug-likeness
3. **Direction**: Whether higher or lower is better
4. **Fitness Calculation**: GA uses these to compute fitness scores

### Example: Optimizing to Config Ranges

```python
# Using preferred ranges from config
molecules = gen.optimize(
    target_properties={
        'logp': 2.0,   # Within preferred [1, 3]
        'qed': 0.95,   # High = better
        'sa': 2.5,     # Low = easier synthesis
        'tpsa': 75     # Optimal for permeability
    }
)
```

---

**Last Updated:** October 14, 2025

