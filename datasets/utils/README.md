# Dataset Processing Utilities

This package contains common molecular processing functions shared between GuacaMol, MOSES, and other dataset processing scripts.

## Functions

### Core Processing
- **`smiles_to_3d_structure()`** - Convert SMILES to 3D coordinates using RDKit
- **`process_smiles_to_npz()`** - Process SMILES file to NPZ format
- **`read_smiles_file()`** - Read SMILES from .smiles or .csv files

### Data Transformation  
- **`remove_hydrogens_from_npz()`** - Remove hydrogen atoms from NPZ files
- **`create_mol2_samples()`** - Generate MOL2 sample files for visualization

### Utilities
- **`check_dependencies()`** - Check if RDKit, OpenBabel, Pandas are available
- **`print_dependency_status()`** - Print dependency status with install instructions

## Usage

```python
from utils.molecular_processing import process_smiles_to_npz, remove_hydrogens_from_npz

# Convert SMILES to NPZ
process_smiles_to_npz("data.smiles", "output.npz", test_mode=True)

# Remove hydrogens
remove_hydrogens_from_npz("with_h.npz", "without_h.npz")
```

## Dependencies

- **RDKit**: SMILES parsing and 3D coordinate generation
- **OpenBabel**: MOL2 file generation and bond perception  
- **Pandas**: CSV file reading
- **NumPy**: Array operations and NPZ file handling
- **tqdm**: Progress bars

## Benefits

- **Code Reuse**: Eliminates duplicate functions between datasets
- **Consistency**: Same 3D generation parameters across all datasets
- **Maintainability**: Single place to update core logic
- **Testing**: Centralized functions are easier to test and debug
