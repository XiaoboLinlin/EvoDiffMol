import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

logger = logging.getLogger(__name__)

def fix_mol2_radicals_and_charges(mol, original_smiles=None):
    """
    Fix problematic MOL2 parsing issues with radicals and formal charges.
    
    This function addresses RDKit's interpretation of MOL2 files where aromatic
    heterocycles may be parsed with incorrect radicals or formal charges.
    
    Args:
        mol: RDKit molecule object (will be copied, original unchanged)
        original_smiles: Original SMILES string (optional, for logging)
        
    Returns:
        str: Fixed canonical SMILES string, or original if fix unsuccessful
    """
    from utils.reconstruct import mol2smiles
    
    # Get original SMILES if not provided
    if original_smiles is None:
        original_smiles = mol2smiles(mol)
        if original_smiles is None:
            original_smiles = Chem.MolToSmiles(mol)
    
    # Check if fixing is needed
    if not original_smiles or not ('[' in original_smiles or '-' in original_smiles or '+' in original_smiles):
        return original_smiles
    
    logger.warning(f"MOL2 parsed SMILES contains radicals/charges: {original_smiles}")
    logger.info("Attempting to fix by resetting radical electrons and formal charges...")
    
    try:
        # Create a copy and fix radical electrons and formal charges
        fixed_mol = Chem.Mol(mol)
        for atom in fixed_mol.GetAtoms():
            atom.SetNumRadicalElectrons(0)
            atom.SetFormalCharge(0)
        
        # Re-sanitize the molecule
        Chem.SanitizeMol(fixed_mol)
        fixed_smiles = mol2smiles(fixed_mol)
        
        if fixed_smiles and '[' not in fixed_smiles and '-' not in fixed_smiles and '+' not in fixed_smiles:
            logger.info(f"Successfully fixed SMILES: {original_smiles} -> {fixed_smiles}")
            return fixed_smiles
        else:
            logger.warning(f"Fix attempt unsuccessful, using original SMILES: {original_smiles}")
            return original_smiles
            
    except Exception as e:
        logger.warning(f"Failed to fix molecule: {e}, using original SMILES: {original_smiles}")
        return original_smiles

def parse_mol2_fixed_structure(mol2_path, remove_h=True, dataset_info=None):
    """
    Parse MOL2 file and extract fixed structure information
    
    Args:
        mol2_path (str): Path to MOL2 file
        remove_h (bool): Whether to remove hydrogen atoms
        dataset_info (dict, optional): Dataset configuration for atom type mapping
        
    Returns:
        dict: Dictionary containing fixed structure information
    """
    try:
        # Read MOL2 file
        mol = Chem.MolFromMol2File(mol2_path, removeHs=remove_h)
        if mol is None:
            raise ValueError(f"Cannot parse MOL2 file: {mol2_path}")
        
        # Get conformer (3D coordinates)
        conf = mol.GetConformer()
        if conf is None:
            raise ValueError(f"No 3D coordinates found in MOL2 file: {mol2_path}")
        
        # Extract atom positions
        fixed_positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            fixed_positions.append([pos.x, pos.y, pos.z])
        
        fixed_positions = torch.tensor(fixed_positions, dtype=torch.float32)
        
        # Extract atom types - convert atomic numbers to dataset encoding
        fixed_atom_types = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            
            # Convert atomic number to dataset atom type encoding
            if dataset_info and 'atom_index' in dataset_info:
                # Use dataset's atom_index mapping (atomic_num -> dataset_type)
                if atomic_num in dataset_info['atom_index']:
                    atom_type = dataset_info['atom_index'][atomic_num]
                else:
                    raise ValueError(f"Atom type {atomic_num} ({atom.GetSymbol()}) not supported in dataset")
            else:
                # Fallback: use atomic number directly (old behavior)
                atom_type = atomic_num
                
            fixed_atom_types.append(atom_type)
        
        fixed_atom_types = torch.tensor(fixed_atom_types, dtype=torch.long)
        
        # Get bond information
        bond_indices = []
        bond_types = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_indices.extend([[i, j], [j, i]])  # Undirected graph
            bond_type = int(bond.GetBondType())
            bond_types.extend([bond_type, bond_type])
        
        bond_index = torch.tensor(bond_indices, dtype=torch.long).t() if bond_indices else torch.empty((2, 0), dtype=torch.long)
        bond_type = torch.tensor(bond_types, dtype=torch.long) if bond_types else torch.empty((0,), dtype=torch.long)
        
        # Generate SMILES for reference (use canonical SMILES with radicals removed)
        try:
            from utils.reconstruct import mol2smiles
            smiles = mol2smiles(mol)
            if smiles is None:
                # Fallback to standard conversion if mol2smiles fails
                smiles = Chem.MolToSmiles(mol)
            
            # Fix problematic SMILES with radicals or charges using centralized function
            smiles = fix_mol2_radicals_and_charges(mol, smiles)
                    
        except:
            smiles = None
            logger.warning(f"Could not generate SMILES for molecule in {mol2_path}")
        
        fixed_structure = {
            'fixed_positions': fixed_positions,
            'fixed_atom_types': fixed_atom_types,
            'bond_index': bond_index,
            'bond_type': bond_type,
            'fixed_mol': mol,
            'num_fixed_atoms': mol.GetNumAtoms(),
            'smiles': smiles
        }
        
        logger.info(f"Loaded fixed structure from {mol2_path}:")
        logger.info(f"  - Number of atoms: {fixed_structure['num_fixed_atoms']}")
        logger.info(f"  - Number of bonds: {len(bond_types) // 2}")
        logger.info(f"  - SMILES: {smiles}")
        
        return fixed_structure
        
    except Exception as e:
        raise ValueError(f"Error parsing MOL2 file {mol2_path}: {str(e)}")

def create_update_mask(num_fixed_atoms, target_total_atoms):
    """
    Create update mask for fixed structure generation
    
    Args:
        num_fixed_atoms (int): Number of fixed atoms
        target_total_atoms (int): Target total number of atoms
        
    Returns:
        torch.Tensor: Update mask (0=fixed, 1=updateable)
    """
    if target_total_atoms < num_fixed_atoms:
        raise ValueError(f"Target total atoms ({target_total_atoms}) must be >= fixed atoms ({num_fixed_atoms})")
    
    num_new_atoms = target_total_atoms - num_fixed_atoms
    
    # Create update mask: 0 for fixed atoms, 1 for new atoms
    update_mask = torch.cat([
        torch.zeros(num_fixed_atoms, dtype=torch.float32),  # Fixed atoms
        torch.ones(num_new_atoms, dtype=torch.float32)      # New atoms
    ])
    
    return update_mask

def validate_fixed_structure(fixed_structure, dataset_info):
    """
    Validate that fixed structure is compatible with dataset
    
    Args:
        fixed_structure (dict): Fixed structure information
        dataset_info (dict): Dataset configuration
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    # Check atom types are valid for dataset
    valid_atom_types = set(dataset_info['atom_types'])
    fixed_atom_types_set = set(fixed_structure['fixed_atom_types'].tolist())
    
    invalid_types = fixed_atom_types_set - valid_atom_types
    if invalid_types:
        raise ValueError(f"Fixed structure contains invalid atom types {invalid_types}. "
                        f"Valid types for {dataset_info.get('name', 'dataset')}: {valid_atom_types}")
    
    # Check reasonable size
    max_atoms = dataset_info.get('max_num_atoms', 50)
    if fixed_structure['num_fixed_atoms'] > max_atoms:
        logger.warning(f"Fixed structure has {fixed_structure['num_fixed_atoms']} atoms, "
                      f"which exceeds typical dataset maximum of {max_atoms}")
    
    return True 