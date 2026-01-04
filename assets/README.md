# EvoDiffMol Assets

This folder contains configuration files and instructions for downloading the pre-trained model checkpoint.

## 📂 Contents

### Configs (Included in Repo)
```
assets/configs/
├── general_without_h.yml      # Model architecture config
├── moses_production.yml        # GA config with property definitions
└── datasets_config.py          # Dataset configurations
```

### Checkpoint (Download from Hugging Face)
```
assets/checkpoints/
└── moses_without_h_80.pt       # Pre-trained model (download required)
```

---

## 🚀 Quick Start

### 1. Download Pre-trained Checkpoint

**Option A: Using Python script (Recommended)**
```bash
python assets/download_checkpoint.py
```

**Option B: Manual Download**
```bash
# Download from Hugging Face
wget https://huggingface.co/YOUR_USERNAME/EvoDiffMol/resolve/main/moses_without_h_80.pt \
     -O assets/checkpoints/moses_without_h_80.pt
```

**Option C: Using huggingface-cli**
```bash
pip install huggingface-hub
huggingface-cli download YOUR_USERNAME/EvoDiffMol moses_without_h_80.pt \
    --local-dir assets/checkpoints
```

---

## 💡 Usage Example

After downloading the checkpoint:

```python
from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

# Load dataset (for metadata)
dataset = General3D('moses', split='valid', remove_h=True)

# Initialize generator with assets configs
gen = MoleculeGenerator(
    checkpoint_path="assets/checkpoints/moses_without_h_80.pt",
    model_config="assets/configs/general_without_h.yml",
    ga_config="assets/configs/moses_production.yml",
    dataset=dataset
)

# Optimize molecules
molecules = gen.optimize(
    target_properties={'qed': 0.9, 'logp': 2.5},
    population_size=100,
    generations=20
)

print(f"Generated {len(molecules)} optimized molecules!")
```

---

## 📊 Checkpoint Details

**Model:** MOSES without H (trained on MOSES dataset with hydrogens removed)
- **File:** `moses_without_h_80.pt`
- **Size:** ~XXX MB (will be specified after upload)
- **Training:** 80 epochs on MOSES dataset
- **Properties:** Supports LogP, QED, SA, TPSA, and 40+ ADMET properties
- **License:** MIT

---

## 🔧 Alternative: Use Your Own Checkpoint

If you have trained your own model:

```python
gen = MoleculeGenerator(
    checkpoint_path="path/to/your/checkpoint.pt",
    model_config="assets/configs/general_without_h.yml",  # Use our configs
    ga_config="assets/configs/moses_production.yml",
    dataset=dataset
)
```

---

## 📝 Notes

- The checkpoint file (~XXX MB) is hosted on Hugging Face to keep the GitHub repo lightweight
- First-time users: Run the download script before using the package
- The configs are small and included in the repo for convenience
- See the main README.md for complete installation and usage instructions

