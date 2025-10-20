# Runtime Information

## CPU Core Usage

Your system has **32 CPU cores** available.

### How EvoDiffMol Uses CPUs

1. **Data Loading (PyTorch DataLoader)**
   - Default: Uses multiple worker processes for data loading
   - Typically uses 4-8 cores for parallel data loading
   - Configured in `evodiffmol/ga/core/trainer.py`

2. **RDKit Calculations**
   - Property calculations (QED, LogP, SA, TPSA) run on CPU
   - Some operations are parallelized automatically

3. **Model Inference**
   - Main computation happens on GPU
   - CPU only used for preprocessing/postprocessing

### Example Runtimes

With **population_size=32** and **generations=3**:

| Example | Approx Runtime | Notes |
|---------|---------------|-------|
| 01_single_property.py | ~2-3 min | Single property (QED) |
| 02_multi_property.py | ~2-3 min | Three properties (LogP, QED, SA) |
| 03_dataframe_output.py | ~2-3 min | Same as #2, returns DataFrame |
| 04_scaffold_based.py | ~3-4 min | Scaffold constraint adds filtering |
| 05_property_exploration.py | ~6-8 min | Runs 3 optimization jobs |
| 06_batch_processing.py | ~10-12 min | Runs 4 optimization jobs |

### Performance Tips

**To use more CPU cores for data loading:**

Edit `evodiffmol/ga/core/trainer.py` and increase `num_workers`:

```python
data_loader = DataLoader(
    dataset,
    batch_size=self.config.batch_size,
    shuffle=True,
    num_workers=8,  # Increase from default (usually 4)
    pin_memory=True
)
```

**Current Configuration:**
- Population size: 32 molecules
- Generations: 3 iterations
- Batch size: 32 (from config, or 128 for production)
- GPU: CUDA enabled
- CPU cores: 32 available

### Monitoring CPU Usage

While running an example:

```bash
# Terminal 1: Run example
python examples/01_single_property.py

# Terminal 2: Monitor CPU usage
htop
# or
top
```

### Scaling for Production

For production runs, increase:

```python
gen.optimize(
    target_properties={'qed': 0.9},
    population_size=200,  # More molecules
    generations=30,       # More iterations  
    batch_size=128,       # Larger batches (if GPU allows)
)
```

**Production runtime estimates:**
- population_size=200, generations=30: ~30-60 minutes
- population_size=500, generations=50: ~2-4 hours

