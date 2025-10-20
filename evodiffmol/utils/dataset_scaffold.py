"""
Scaffold-based dataset for molecular generation with fixed scaffold structures.
"""

import torch
import logging
import numpy as np
import os.path as osp
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data, InMemoryDataset
from utils.reconstruct import mol2smiles
from utils.mol2_parser import create_update_mask

logger = logging.getLogger(__name__)


def get_scaffold_from_mol(mol):
    """
    Extract the scaffold (core structure) from a molecule using RDKit.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        str: SMILES of the scaffold, or None if extraction fails
    """
    if mol is None:
        return None
    
    # Use Murcko scaffold to get the core structure
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is not None:
        return Chem.MolToSmiles(scaffold)
    return None


class GeneralScaffoldDataset(InMemoryDataset):
    """
    General scaffold dataset that can work with any NPZ format dataset (QM40, MOSES, GuacaMol, etc.)
    filtered to contain only molecules with a specific scaffold.
    """
    
    def __init__(self, scaffold_mol2_path: str, dataset_name: str, root='./ga_output', split='train', 
                 min_molecules: int = 200, max_molecules: int = 10000,
                 remove_h: bool = True, transform=None, pre_transform=None, pre_filter=None, 
                 val_split_ratio: float = 0.2, project_name: str = None):
        """
        Initialize scaffold dataset.
        
        Args:
            scaffold_mol2_path: Path to MOL2 file containing the scaffold structure
            dataset_name: Name of the dataset (e.g., 'qm40', 'moses', 'guacamol')
            root: Root directory for dataset
            split: Dataset split ('train', 'valid', 'test')
            min_molecules: Minimum number of molecules required
            max_molecules: Maximum number of molecules to include
            remove_h: Whether hydrogens are removed from the dataset
            transform, pre_transform, pre_filter: Standard PyTorch Geometric transforms
            project_name: Optional project name for cleaner directory naming (e.g., 'moses_pyridine')
        """
        self.scaffold_mol2_path = scaffold_mol2_path
        self.dataset_name = dataset_name.upper()  # Normalize to uppercase for consistency
        self.split = split
        self.min_molecules = min_molecules
        self.max_molecules = max_molecules
        self.remove_h = remove_h
        self.val_split_ratio = val_split_ratio
        
        # Choose directory naming based on whether project_name is provided
        if project_name:
            # Use clean project-based naming
            self.folder = osp.join(root, project_name)
            if remove_h:
                self.processed_file_suffix = 'scaffold_noh'
            else:
                self.processed_file_suffix = 'scaffold_h'
        else:
            # Fallback to naming based on dataset and scaffold
            scaffold_name = osp.basename(scaffold_mol2_path).replace('.mol2', '')
            if remove_h:
                self.folder = osp.join(root, f'{dataset_name.lower()}_noh_{scaffold_name}')
                self.processed_file_suffix = f'{dataset_name.lower()}_noh'
            else:
                self.folder = osp.join(root, f'{dataset_name.lower()}_h_{scaffold_name}')
                self.processed_file_suffix = f'{dataset_name.lower()}_h'
            
        # Load and parse scaffold
        self.target_scaffold_smiles = self._load_scaffold()
        logger.info(f"Target scaffold SMILES: {self.target_scaffold_smiles}")
        print(f"[DATASET] Target scaffold SMILES: {self.target_scaffold_smiles}")
        logger.info(f"Dataset folder: {self.folder}")
        print(f"[DATASET] Dataset folder: {self.folder}")
        logger.info(f"Using dataset: {self.dataset_name}")
        print(f"[DATASET] Using dataset: {self.dataset_name}")
        
        super(GeneralScaffoldDataset, self).__init__(self.folder, transform, pre_transform, pre_filter)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _load_scaffold(self) -> str:
        """Load scaffold MOL2 file and extract SMILES."""
        scaffold_mol = Chem.MolFromMol2File(self.scaffold_mol2_path, removeHs=self.remove_h)
        if scaffold_mol is None:
            raise ValueError(f"Cannot parse MOL2 file: {self.scaffold_mol2_path}")

        target_scaffold_smiles = mol2smiles(scaffold_mol)
        if target_scaffold_smiles is None:
            raise ValueError(f"Cannot generate SMILES from scaffold in {self.scaffold_mol2_path}")

        # Fix problematic MOL2 parsing cases with radicals or charges using centralized function
        from utils.mol2_parser import fix_mol2_radicals_and_charges
        return fix_mol2_radicals_and_charges(scaffold_mol, target_scaffold_smiles)

    def _get_scaffold_mol(self):
        """Get the scaffold molecule (cached)."""
        if not hasattr(self, '_scaffold_mol_cache'):
            self._scaffold_mol_cache = Chem.MolFromMol2File(self.scaffold_mol2_path, removeHs=self.remove_h)
        return self._scaffold_mol_cache

    def _get_molecule_from_smiles(self, smiles: str):
        """Convert SMILES string to RDKit molecule."""
        return Chem.MolFromSmiles(smiles)

    def _load_smiles_from_csv(self) -> np.ndarray:
        """Load SMILES from original CSV file for datasets that don't include them in NPZ."""
        import pandas as pd
        
        # Build path to original CSV file with dataset-specific naming conventions
        if self.dataset_name.upper() == 'MOSES' and self.split == 'valid':
            # MOSES uses test_scaffolds.csv for validation split
            csv_file = osp.join('./datasets', self.dataset_name.lower(), 'original_csv', 'test_scaffolds.csv')
        else:
            # Standard naming convention
            csv_file = osp.join('./datasets', self.dataset_name.lower(), 'original_csv', f'{self.split}.csv')
        
        if not osp.exists(csv_file):
            raise FileNotFoundError(f"Original CSV file not found: {csv_file}")
        
        logger.info(f"Loading SMILES from: {csv_file}")
        print(f"[DATASET] Loading SMILES from: {csv_file}")
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Extract SMILES column (assume it's named 'SMILES' or 'smiles')
        if 'SMILES' in df.columns:
            smiles_array = df['SMILES'].values
        elif 'smiles' in df.columns:
            smiles_array = df['smiles'].values
        else:
            raise ValueError(f"No SMILES column found in {csv_file}. Available columns: {list(df.columns)}")
        
        return smiles_array

    def _load_and_filter_scaffold_data(self):
        """Load dataset data and filter for molecules containing the scaffold."""
        logger.info(f"Loading and filtering {self.dataset_name} data for scaffold...")
        print(f"[DATASET] Loading and filtering {self.dataset_name} data for scaffold...")
        
        # Build path based on dataset name and remove_h flag
        # Handle different dataset directory structures
        if self.dataset_name.upper() == 'QM40':
            # QM40 has structure: datasets/QM40/qm40_without_h/raw/
            if self.remove_h:
                raw_dir = osp.join('./datasets', 'QM40', 'qm40_without_h', 'raw')
            else:
                raw_dir = osp.join('./datasets', 'QM40', 'qm40_with_h', 'raw')
        else:
            # Other datasets (MOSES, GuacaMol) have structure: datasets/moses/without_h/raw/
            if self.remove_h:
                raw_dir = osp.join('./datasets', self.dataset_name.lower(), 'without_h', 'raw')
            else:
                raw_dir = osp.join('./datasets', self.dataset_name.lower(), 'with_h', 'raw')
        
        npz_file = osp.join(raw_dir, f'{self.split}.npz')
        print(f"[DATASET] Loading from: {npz_file}")
        
        if not osp.exists(npz_file):
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")
            
        data = np.load(npz_file, allow_pickle=True)
        
        # Check if SMILES are available in NPZ file
        if 'smiles' in data:
            # QM40 case: SMILES are in the NPZ file
            smiles = data['smiles']
            logger.info(f"Loaded {len(smiles)} molecules from {npz_file}")
            print(f"[DATASET] Loaded {len(smiles)} molecules from {npz_file}")
        else:
            # MOSES/GuacaMol case: Load SMILES from original CSV file
            logger.info(f"SMILES not found in NPZ file, loading from original CSV...")
            print(f"[DATASET] SMILES not found in NPZ file, loading from original CSV...")
            
            smiles = self._load_smiles_from_csv()
            logger.info(f"Loaded {len(smiles)} SMILES from CSV and {len(data['num_atoms'])} molecules from NPZ")
            print(f"[DATASET] Loaded {len(smiles)} SMILES from CSV and {len(data['num_atoms'])} molecules from NPZ")
        
        # Extract data
        R = data['positions']  # (N, max_atoms, 3)
        Z = data['charges']    # (N, max_atoms)  
        N = data['num_atoms']  # (N,)
        
        # Find molecules containing the scaffold
        matching_indices = self._find_scaffold_matches(smiles)
        
        # Check minimum requirement
        if len(matching_indices) < self.min_molecules:
            raise ValueError(f"Found only {len(matching_indices)} molecules with scaffold, "
                            f"but need at least {self.min_molecules}")
        
        logger.info(f"Found {len(matching_indices)} molecules containing scaffold")
        print(f"[DATASET] Found {len(matching_indices)} molecules containing scaffold")
        
        # Filter data to matching molecules
        filtered_data = []
        for mol_idx in tqdm(matching_indices, desc="Creating filtered dataset"):
            n_atoms = int(N[mol_idx])
            positions = torch.tensor(R[mol_idx][:n_atoms, :], dtype=torch.float32)
            atom_types = torch.tensor(Z[mol_idx][:n_atoms], dtype=torch.long)
            mol_smiles = smiles[mol_idx]
            
            # Center the molecular positions
            positions = positions - positions.mean(dim=0, keepdim=True)
            
            # Create update mask for scaffold training
            update_mask = self._create_update_mask(positions, atom_types, mol_smiles)
            
            # Reorder atoms: scaffold atoms (0) first, then variable atoms (1)
            scaffold_indices = torch.where(update_mask == 0.0)[0]
            variable_indices = torch.where(update_mask == 1.0)[0]
            
            # Create reordering indices: scaffold first, then variable
            reorder_indices = torch.cat([scaffold_indices, variable_indices])
            
            # Reorder positions, atom_types, and update_mask
            positions_reordered = positions[reorder_indices]
            atom_types_reordered = atom_types[reorder_indices]
            update_mask_reordered = update_mask[reorder_indices]
            
            # Verify the reordering worked correctly
            num_scaffold = len(scaffold_indices)
            assert torch.all(update_mask_reordered[:num_scaffold] == 0.0), "Scaffold atoms should be first"
            assert torch.all(update_mask_reordered[num_scaffold:] == 1.0), "Variable atoms should be last"
            
            # Create Data object with reordered data
            data_obj = Data(
                pos=positions_reordered,
                atom_type=atom_types_reordered,
                smiles=mol_smiles,
                num_atoms=torch.tensor(n_atoms),
                update_mask=update_mask_reordered,
                num_scaffold_atoms=torch.tensor(num_scaffold)  # Store for reference
            )
            
            filtered_data.append(data_obj)
        
        logger.info(f"Created scaffold dataset with {len(filtered_data)} molecules")
        return filtered_data

    def _find_scaffold_matches(self, smiles_array: np.ndarray) -> List[int]:
        """Find molecules that contain the target scaffold as a substructure."""
        matching_indices = []
        
        # Use SMILES-based scaffold molecule for matching (not MOL2)
        target_scaffold_mol = self._get_molecule_from_smiles(self.target_scaffold_smiles)
        
        if target_scaffold_mol is None:
            logger.error(f"Invalid target scaffold SMILES: {self.target_scaffold_smiles}")
            print(f"[DATASET ERROR] Invalid target scaffold SMILES: {self.target_scaffold_smiles}")
            return matching_indices
        
        logger.info(f"Searching for molecules containing scaffold substructure: {self.target_scaffold_smiles}")
        logger.info(f"Will stop after finding {self.max_molecules} matching molecules")
        print(f"[DATASET] Searching for molecules containing scaffold: {self.target_scaffold_smiles}")
        print(f"[DATASET] Will stop after finding {self.max_molecules} matching molecules")
        
        failed_count = 0
        for i, smiles_str in enumerate(tqdm(smiles_array, desc="Filtering molecules with scaffold")):
            if smiles_str is None:
                failed_count += 1
                continue
                
            # Convert SMILES to molecule
            mol = self._get_molecule_from_smiles(smiles_str)
            if mol is None:
                failed_count += 1
                continue
                
            # Check if molecule contains the target scaffold as a substructure
            if mol.HasSubstructMatch(target_scaffold_mol):
                matching_indices.append(i)
                
                # Stop early if we have enough molecules
                if len(matching_indices) >= self.max_molecules:
                    logger.info(f"Found {len(matching_indices)} matches (target: {self.max_molecules}), stopping search early")
                    print(f"[DATASET] Found {len(matching_indices)} matches (target: {self.max_molecules}), stopping search early")
                    break
                
            # Log progress
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(smiles_array)} molecules, "
                           f"found {len(matching_indices)} matches, "
                           f"{failed_count} failed SMILES conversions")
                print(f"[DATASET] Processed {i + 1}/{len(smiles_array)} molecules, "
                      f"found {len(matching_indices)} matches, "
                      f"{failed_count} failed SMILES conversions")
        
        logger.info(f"Scaffold filtering complete: {len(matching_indices)} matches, {failed_count} failures")
        print(f"[DATASET] Scaffold filtering complete: {len(matching_indices)} matches, {failed_count} failures")
        return matching_indices

    def _create_update_mask(self, positions: torch.Tensor, atom_types: torch.Tensor, 
                          smiles: str) -> torch.Tensor:
        """
        Create update mask for scaffold-based training using proper substructure matching.
        
        Args:
            positions: Atom positions (N, 3)
            atom_types: Atom types (N,)
            smiles: Molecule SMILES
            
        Returns:
            torch.Tensor: Update mask (N,) where 0=fixed scaffold atoms, 1=variable atoms
        """
        # Convert molecule SMILES to RDKit molecule
        mol = self._get_molecule_from_smiles(smiles)
        if mol is None:
            # Fallback: mark all atoms as updateable if SMILES parsing fails
            return torch.ones(len(atom_types), dtype=torch.float32)
        
        # Get scaffold molecule using SMILES (same as _find_scaffold_matches)
        scaffold_mol = self._get_molecule_from_smiles(self.target_scaffold_smiles)
        if scaffold_mol is None:
            # Fallback: mark all atoms as updateable if scaffold loading fails
            return torch.ones(len(atom_types), dtype=torch.float32)
        
        num_total_atoms = len(atom_types)
        update_mask = torch.ones(num_total_atoms, dtype=torch.float32)
        
        # Find all substructure matches
        matches = mol.GetSubstructMatches(scaffold_mol)
        if matches:
            if len(matches) > 1:
                # Multiple scaffold instances found - randomly pick one
                import random
                selected_match_idx = random.randint(0, len(matches) - 1)
                scaffold_atom_indices = matches[selected_match_idx]
                logger.debug(f"Found {len(matches)} scaffold instances, randomly selected instance {selected_match_idx}")
            else:
                # Only one scaffold instance
                scaffold_atom_indices = matches[0]
            
            # Mark scaffold atoms as fixed (0)
            for atom_idx in scaffold_atom_indices:
                if atom_idx < num_total_atoms:  # Safety check
                    update_mask[atom_idx] = 0.0
            
            # Log the mapping for debugging
            num_fixed = len(scaffold_atom_indices)
            num_variable = num_total_atoms - num_fixed
            logger.debug(f"Scaffold mapping: {num_fixed} fixed atoms, {num_variable} variable atoms")
        else:
            # This shouldn't happen since we already filtered for scaffold-containing molecules
            logger.warning(f"No scaffold match found in molecule {smiles}, marking all atoms as variable")
        
        return update_mask

    @property  
    def raw_file_names(self):
        return [f'{self.split}.npz']

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def download(self):
        # No download needed - we use the existing QM40 raw data
        pass

    def process(self):
        # Ensure processed directory exists
        import os
        processed_dir = osp.dirname(self.processed_paths[0])
        os.makedirs(processed_dir, exist_ok=True)
        
        # Load and filter data
        filtered_data = self._load_and_filter_scaffold_data()
        
        # Apply pre_filter if specified
        if self.pre_filter is not None:
            filtered_data = [data for data in filtered_data if self.pre_filter(data)]
        
        # Apply pre_transform if specified
        if self.pre_transform is not None:
            filtered_data = [self.pre_transform(data) for data in filtered_data]
        
        # Collate data for efficient batching
        self.data, self.slices = self.collate(filtered_data)
        
        # Save processed data
        torch.save((self.data, self.slices), self.processed_paths[0])
        logger.info(f"Processed data saved to {self.processed_paths[0]}")

    def get_scaffold_info(self) -> Dict[str, Any]:
        """Get information about the scaffold."""
        scaffold_mol = Chem.MolFromMol2File(self.scaffold_mol2_path, removeHs=self.remove_h)
        return {
            'smiles': self.target_scaffold_smiles,
            'num_atoms': scaffold_mol.GetNumAtoms(),
            'mol2_path': self.scaffold_mol2_path
        }


# Backward compatibility alias for existing code
class QM40ScaffoldDataset(GeneralScaffoldDataset):
    """
    Backward compatibility wrapper for QM40-specific scaffold dataset.
    Automatically sets dataset_name='qm40'.
    """
    
    def __init__(self, scaffold_mol2_path: str, root='./ga_output', split='train', 
                 min_molecules: int = 200, max_molecules: int = 10000,
                 remove_h: bool = True, transform=None, pre_transform=None, pre_filter=None, 
                 val_split_ratio: float = 0.2, project_name: str = None):
        super().__init__(
            scaffold_mol2_path=scaffold_mol2_path,
            dataset_name='qm40',
            root=root,
            split=split,
            min_molecules=min_molecules,
            max_molecules=max_molecules,
            remove_h=remove_h,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            val_split_ratio=val_split_ratio,
            project_name=project_name
        ) 