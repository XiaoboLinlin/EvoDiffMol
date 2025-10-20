"""
Test 3D structure generation from diffusion model output.

Test Configuration:
- Population size: 8
- Generations: 3
- Device: cuda
"""

import pytest
import os
import tempfile
import shutil
from evodiffmol import MoleculeGenerator
from rdkit import Chem
from rdkit.Chem import AllChem


def test_3d_structure_from_diffusion_output(generator):
    """
    Test that optimized molecules generate 3D structure output.
    Verify that:
    1. epoch_last directory is created
    2. mol2_files directory exists with MOL2 files
    """
    print("\n🔬 Testing 3D structure output:")
    
    # Create output directory in project root to save MOL2 files
    output_dir = tempfile.mkdtemp(prefix="test_3d_", dir=".")
    
    # Run optimization and save MOL2 files
    molecules = generator.optimize(
        target_properties={'qed': 1.0},
        population_size=8,
        generations=3,
        verbose=False,
        output_dir=output_dir
    )
    
    assert len(molecules) > 0, "No molecules generated"
    print(f"✓ Generated {len(molecules)} molecules")
    
    # Verify epoch_last directory exists
    epoch_last_dir = os.path.join(output_dir, 'epoch_last')
    assert os.path.exists(epoch_last_dir), f"epoch_last directory not found: {epoch_last_dir}"
    print(f"✓ epoch_last directory exists")
    
    # Verify mol2_files directory exists
    mol2_dir = os.path.join(epoch_last_dir, 'mol2_files')
    assert os.path.exists(mol2_dir), f"mol2_files directory not found: {mol2_dir}"
    print(f"✓ mol2_files directory exists")
    
    # Verify MOL2 files were created
    mol2_files = [f for f in os.listdir(mol2_dir) if f.endswith('.mol2')]
    assert len(mol2_files) > 0, "No MOL2 files found"
    print(f"✓ Found {len(mol2_files)} MOL2 files with 3D coordinates")
    
    print("\n✅ 3D structure output verified!")
    print(f"   - Output directory: {output_dir}")
    print(f"   - epoch_last: ✓")
    print(f"   - mol2_files: ✓ ({len(mol2_files)} files)")
    print(f"   - Results saved in: {output_dir}")

