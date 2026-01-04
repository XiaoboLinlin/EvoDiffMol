"""
Enhanced scaffold dataset that accepts SMILES instead of MOL2 files.

This is a convenience wrapper for GeneralScaffoldDataset that allows
specifying scaffolds using SMILES strings directly.
"""

import os
import tempfile
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.dataset_scaffold import GeneralScaffoldDataset


def create_scaffold_dataset_from_smiles(
    scaffold_smiles: str,
    dataset_name: str = 'moses',
    split: str = 'train',
    min_molecules: int = 200,
    max_molecules: int = 10000,
    remove_h: bool = True,
    root: str = './ga_output',
    project_name: Optional[str] = None,
    pre_transform = None  # ← ADD: PyG pre_transform for compatibility
) -> GeneralScaffoldDataset:
    """
    Create a scaffold dataset using just a SMILES string (no MOL2 file needed).
    
    This is a convenience function that:
    1. Converts SMILES to RDKit molecule
    2. Generates 3D coordinates
    3. Saves temporary MOL2 file
    4. Creates GeneralScaffoldDataset
    
    Args:
        scaffold_smiles: SMILES string of the scaffold (e.g., 'c1ccccc1' for benzene)
        dataset_name: Name of dataset to filter ('moses', 'qm40', etc.)
        split: Dataset split ('train', 'valid', 'test')
        min_molecules: Minimum number of matching molecules required
        max_molecules: Maximum number of molecules to include
        remove_h: Whether hydrogens are removed
        root: Root directory for output
        project_name: Optional project name for directory naming
        pre_transform: PyTorch Geometric pre_transform (e.g., AtomFeat) for creating required attributes
        
    Returns:
        GeneralScaffoldDataset instance
        
    Example:
        >>> # Simple API - just provide SMILES!
        >>> scaffold_dataset = create_scaffold_dataset_from_smiles(
        ...     scaffold_smiles='c1ccccc1',  # Benzene
        ...     dataset_name='moses',
        ...     max_molecules=10000
        ... )
        >>> # Use with MoleculeGenerator
        >>> gen = MoleculeGenerator(checkpoint, dataset=scaffold_dataset)
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {scaffold_smiles}")
    
    # Add hydrogens if needed (for 3D generation)
    if not remove_h:
        mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Create temporary MOL2 file
    # Use a persistent temp file in the project directory
    temp_dir = os.path.join(root, '.temp_scaffolds')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate filename based on SMILES (sanitized)
    scaffold_name = scaffold_smiles.replace('/', '_').replace('\\', '_')[:50]
    temp_mol2_path = os.path.join(temp_dir, f'scaffold_{scaffold_name}.mol2')
    
    # Save to MOL2 using proper MOL2 writer
    from utils.mol2_writer import write_mol2_structure
    
    # Get SMILES and positions
    smiles = Chem.MolToSmiles(mol)
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    
    # Write MOL2 file - signature: write_mol2_structure(mol, pos, smile, output_path, update_mask)
    write_mol2_structure(mol, positions, smiles, temp_mol2_path, update_mask=None)
    
    # If pre_transform is provided, clean cache to force reprocessing
    # (Required for fine-tuning to work with PyTorch Geometric 2.7.0)
    if pre_transform is not None:
        import shutil
        # Determine cache folder name
        scaffold_name = scaffold_smiles.replace('/', '_').replace('\\', '_')[:50]
        dataset_folder = f"{dataset_name.lower()}_{'noh' if remove_h else 'withh'}_scaffold_{scaffold_name}"
        cache_dir = os.path.join(root, dataset_folder)
        processed_dir = os.path.join(cache_dir, 'processed')
        
        # Remove processed cache if it exists
        if os.path.exists(processed_dir):
            print(f"   🔄 Clearing cached dataset (required for pre_transform): {processed_dir}")
            shutil.rmtree(processed_dir)
    
    # Create scaffold dataset using the temporary MOL2 file
    scaffold_dataset = GeneralScaffoldDataset(
        scaffold_mol2_path=temp_mol2_path,
        dataset_name=dataset_name,
        root=root,
        split=split,
        min_molecules=min_molecules,
        max_molecules=max_molecules,
        remove_h=remove_h,
        project_name=project_name,
        pre_transform=pre_transform  # ← FIX: Pass pre_transform for PyG compatibility
    )
    
    return scaffold_dataset


# Convenience function for the API
def create_scaffold_dataset(
    scaffold: str,
    dataset_name: str = 'moses',
    max_molecules: int = 10000,
    **kwargs
) -> GeneralScaffoldDataset:
    """
    Create scaffold dataset from SMILES string (simplified API).
    
    Args:
        scaffold: SMILES string of scaffold
        dataset_name: Dataset to filter
        max_molecules: Maximum molecules to find
        **kwargs: Additional arguments for create_scaffold_dataset_from_smiles
        
    Returns:
        GeneralScaffoldDataset instance
        
    Example:
        >>> # Super simple!
        >>> dataset = create_scaffold_dataset('c1ccccc1', dataset_name='moses')
        >>> gen = MoleculeGenerator(checkpoint, dataset=dataset)
    """
    return create_scaffold_dataset_from_smiles(
        scaffold_smiles=scaffold,
        dataset_name=dataset_name,
        max_molecules=max_molecules,
        **kwargs
    )


if __name__ == '__main__':
    """Test the SMILES-based scaffold dataset creation."""
    
    print("Testing SMILES-based scaffold dataset creation...")
    print("=" * 60)
    
    # Test with benzene
    scaffold_smiles = 'c1ccccc1'
    print(f"\nCreating scaffold dataset for: {scaffold_smiles}")
    
    try:
        dataset = create_scaffold_dataset(
            scaffold=scaffold_smiles,
            dataset_name='moses',
            max_molecules=100,  # Small number for testing
            split='train'
        )
        
        print(f"✅ Success! Created dataset with {len(dataset)} molecules")
        print(f"Scaffold info: {dataset.get_scaffold_info()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

