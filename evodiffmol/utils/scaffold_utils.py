"""
Scaffold-based molecule filtering and fine-tuning utilities for genetic algorithm.
"""

import torch
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from utils.reconstruct import build_molecule, mol2smiles
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


def molecules_contain_scaffold_smiles(dataset_smiles: List[str], target_scaffold_smiles: str) -> List[int]:
    """
    Find indices of molecules that contain the target scaffold structure using SMILES comparison.
    
    Args:
        dataset_smiles: List of SMILES strings from the dataset
        target_scaffold_smiles: SMILES string of the target scaffold
        
    Returns:
        List of indices of molecules containing the scaffold
    """
    matching_indices = []
    target_scaffold_mol = Chem.MolFromSmiles(target_scaffold_smiles)
    
    if target_scaffold_mol is None:
        logger.error(f"Invalid target scaffold SMILES: {target_scaffold_smiles}")
        return matching_indices
    
    logger.info(f"Searching for molecules containing scaffold: {target_scaffold_smiles}")
    logger.info(f"This may take a few minutes to process {len(dataset_smiles)} molecules...")
    
    # Add counters for progress tracking
    processed_count = 0
    failed_count = 0
    
    for i, smiles in enumerate(tqdm(dataset_smiles, desc="Filtering molecules with scaffold", 
                                  unit="mol", ncols=100, mininterval=1.0)):
        if smiles is None:
            failed_count += 1
            continue
            
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_count += 1
            continue
            
        # Get scaffold from this molecule
        mol_scaffold = get_scaffold_from_mol(mol)
        
        if mol_scaffold == target_scaffold_smiles:
            matching_indices.append(i)
            # Log progress when we find matches
            if len(matching_indices) % 50 == 0:
                logger.info(f"Found {len(matching_indices)} matching molecules so far...")
        
        processed_count += 1
        
        # Log progress every 5000 molecules
        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i + 1}/{len(dataset_smiles)} molecules, "
                       f"found {len(matching_indices)} matches, "
                       f"{failed_count} failed SMILES conversions")
    
    logger.info(f"=== Scaffold Filtering Complete ===")
    logger.info(f"Total molecules processed: {processed_count}")
    logger.info(f"Failed SMILES conversions: {failed_count}")
    logger.info(f"Matching molecules found: {len(matching_indices)}")
    
    return matching_indices


def molecules_contain_scaffold(molecules_data: List[Data], target_scaffold_smiles: str, dataset_info: Dict) -> List[int]:
    """
    Find indices of molecules that contain the target scaffold structure.
    
    Args:
        molecules_data: List of molecular Data objects
        target_scaffold_smiles: SMILES string of the target scaffold
        dataset_info: Dataset configuration
        
    Returns:
        List of indices of molecules containing the scaffold
    """
    matching_indices = []
    target_scaffold_mol = Chem.MolFromSmiles(target_scaffold_smiles)
    
    if target_scaffold_mol is None:
        logger.error(f"Invalid target scaffold SMILES: {target_scaffold_smiles}")
        return matching_indices
    
    logger.info(f"Searching for molecules containing scaffold: {target_scaffold_smiles}")
    logger.info(f"This may take a few minutes to process {len(molecules_data)} molecules...")
    
    # Add counters for progress tracking
    processed_count = 0
    failed_count = 0
    
    for i, data in enumerate(tqdm(molecules_data, desc="Filtering molecules with scaffold", 
                                  unit="mol", ncols=100, mininterval=1.0)):
        # Reconstruct molecule from Data object
        atom_type = torch.argmax(data.atom_feat_full[:, :-1], dim=1)  # Exclude charge dimension
        mol = build_molecule(data.pos, atom_type, dataset_info)
        
        if mol is None:
            failed_count += 1
            continue
            
        # Get scaffold from this molecule
        mol_scaffold = get_scaffold_from_mol(mol)
        
        if mol_scaffold == target_scaffold_smiles:
            matching_indices.append(i)
            # Log progress when we find matches
            if len(matching_indices) % 50 == 0:
                logger.info(f"Found {len(matching_indices)} matching molecules so far...")
        
        processed_count += 1
        
        # Log progress every 5000 molecules
        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i + 1}/{len(molecules_data)} molecules, "
                       f"found {len(matching_indices)} matches, "
                       f"{failed_count} failed reconstructions")
    
    logger.info(f"=== Scaffold Filtering Complete ===")
    logger.info(f"Total molecules processed: {processed_count}")
    logger.info(f"Failed reconstructions: {failed_count}")
    logger.info(f"Found {len(matching_indices)} molecules containing the target scaffold")
    return matching_indices


def filter_molecules_with_scaffold(train_dataset, scaffold_mol2_path: str, dataset_info: Dict, 
                                 min_molecules: int = 200, max_molecules: int = 10000, 
                                 remove_h: bool = True) -> Tuple[List[Data], Dict]:
    """
    Filter training dataset to find molecules containing the specified scaffold.
    
    Args:
        train_dataset: Training dataset
        scaffold_mol2_path: Path to MOL2 file containing the scaffold structure
        dataset_info: Dataset configuration
        min_molecules: Minimum number of molecules required
        max_molecules: Maximum number of molecules to return
        remove_h: Whether hydrogens are removed from the dataset
        
    Returns:
        Tuple[List[Data], Dict]: List of Data objects containing the scaffold and scaffold structure info
        
    Raises:
        ValueError: If insufficient molecules are found
    """
    logger.info(f"Filtering molecules with scaffold from: {scaffold_mol2_path}")
    
    # Load scaffold MOL2 file and extract SMILES
    scaffold_mol = Chem.MolFromMol2File(scaffold_mol2_path, removeHs=remove_h)
    if scaffold_mol is None:
        raise ValueError(f"Cannot parse MOL2 file: {scaffold_mol2_path}")
    
    target_scaffold_smiles = mol2smiles(scaffold_mol)
    
    logger.info(f"Target scaffold SMILES: {target_scaffold_smiles}")
    
    # Extract SMILES from dataset for scaffold comparison
    if hasattr(train_dataset, 'data'):
        # Handle InMemoryDataset - extract SMILES directly
        logger.info(f"Extracting SMILES from {len(train_dataset)} molecules in training dataset...")
        dataset_smiles = []
        for i in range(len(train_dataset)):
            data = train_dataset.get(i)
            if hasattr(data, 'smiles') and data.smiles:
                dataset_smiles.append(data.smiles)
            else:
                dataset_smiles.append(None)  # Mark molecules without SMILES
        logger.info(f"Successfully extracted {sum(1 for s in dataset_smiles if s is not None)} valid SMILES")
    else:
        # Assume it's already a list
        logger.info(f"Extracting SMILES from pre-loaded dataset with {len(train_dataset)} molecules...")
        dataset_smiles = []
        for data in train_dataset:
            if hasattr(data, 'smiles') and data.smiles:
                dataset_smiles.append(data.smiles)
            else:
                dataset_smiles.append(None)  # Mark molecules without SMILES
    
    # Find matching molecules using SMILES comparison
    matching_indices = molecules_contain_scaffold_smiles(dataset_smiles, target_scaffold_smiles)
    
    # Check minimum requirement
    if len(matching_indices) < min_molecules:
        raise ValueError(f"Found only {len(matching_indices)} molecules with scaffold, "
                        f"but need at least {min_molecules}")
    
    # Limit to max_molecules if needed
    if len(matching_indices) > max_molecules:
        # Randomly sample max_molecules
        np.random.shuffle(matching_indices)
        matching_indices = matching_indices[:max_molecules]
        logger.info(f"Randomly selected {max_molecules} molecules from {len(matching_indices)} candidates")
    
    # Extract the matching molecules from the dataset
    filtered_molecules = []
    if hasattr(train_dataset, 'data'):
        # Handle InMemoryDataset
        filtered_molecules = [train_dataset.get(i) for i in matching_indices]
    else:
        # Handle list dataset
        filtered_molecules = [train_dataset[i] for i in matching_indices]
    
    logger.info(f"Successfully filtered {len(filtered_molecules)} molecules containing the scaffold")
    
    # Create minimal scaffold structure info needed for other functions
    scaffold_structure = {
        'smiles': target_scaffold_smiles,
        'num_fixed_atoms': scaffold_mol.GetNumAtoms()
    }
    
    return filtered_molecules, scaffold_structure


def create_scaffold_dataset_with_masks(filtered_molecules: List[Data], scaffold_structure: Dict) -> List[Tuple[Data, torch.Tensor]]:
    """
    Create dataset with update masks for scaffold-based fine-tuning.
    
    Args:
        filtered_molecules: List of molecules containing the scaffold
        scaffold_structure: Scaffold structure information
        
    Returns:
        List of tuples (molecule_data, update_mask)
    """
    scaffold_data_with_masks = []
    num_scaffold_atoms = scaffold_structure['num_fixed_atoms']
    
    logger.info(f"Creating scaffold dataset with update masks for {len(filtered_molecules)} molecules")
    logger.info(f"Scaffold has {num_scaffold_atoms} fixed atoms")
    
    for data in tqdm(filtered_molecules, desc="Creating update masks", 
                     unit="mol", ncols=100):
        num_total_atoms = data.pos.shape[0]
        
        # Create update mask: assume scaffold atoms are at the beginning
        # In practice, you might need more sophisticated matching
        # For now, we'll create a mask where the first num_scaffold_atoms are fixed (0)
        # and the rest are movable (1)
        update_mask = create_update_mask(num_scaffold_atoms, num_total_atoms)
        
        scaffold_data_with_masks.append((data, update_mask))
    
    logger.info(f"Created {len(scaffold_data_with_masks)} scaffold training samples")
    return scaffold_data_with_masks 