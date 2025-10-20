#!/usr/bin/env python3
"""
Common molecular processing utilities for dataset generation.

This module provides shared functions for:
- SMILES to 3D structure conversion
- Hydrogen atom removal  
- MOL2 file generation
- Dataset processing pipelines

Used by both GuacaMol and MOSES dataset processing scripts.
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Allowed atom types for molecular processing
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 16, 17, 35}  # H, C, N, O, F, S, Cl, Br

def process_single_smiles(smiles, allowed_atoms=None, validate_reconstruction=True):
    """
    Wrapper function for multiprocessing - processes a single SMILES string with validation.
    
    Args:
        smiles (str): SMILES string to process
        allowed_atoms (set): Set of allowed atomic numbers for filtering
        validate_reconstruction (bool): If True, validate that 3D structure matches original SMILES
        
    Returns:
        tuple: (positions, charges, num_atoms, processed_smiles) or None if failed/invalid
    """
    result = smiles_to_3d_structure(smiles.strip(), allowed_atoms=allowed_atoms)
    
    if result is None or not validate_reconstruction:
        return result
    
    # Validate: compare original and reconstructed SMILES
    positions, atomic_numbers, num_atoms, processed_smiles = result
    
    # Canonicalize original SMILES for fair comparison
    try:
        from rdkit import Chem
        original_mol = Chem.MolFromSmiles(smiles.strip())
        if original_mol is not None:
            # Remove hydrogens and radicals from original for comparison
            original_mol_no_h = Chem.RemoveHs(original_mol)
            for atom in original_mol_no_h.GetAtoms():
                atom.SetNumRadicalElectrons(0)
            
            canonical_original = Chem.MolToSmiles(original_mol_no_h, canonical=True, isomericSmiles=False)
            
            # Compare canonical forms
            if canonical_original == processed_smiles:
                return result  # Valid: 3D structure matches original
            else:
                # Invalid: 3D reconstruction doesn't match original
                return None
        else:
            # Original SMILES invalid
            return None
    except Exception:
        # Comparison failed - be conservative and reject
        return None

def smiles_to_3d_structure(smiles, max_attempts=5, allowed_atoms=None):
    """
    Convert SMILES string to 3D molecular structure using RDKit.
    
    Args:
        smiles (str): SMILES string
        max_attempts (int): Maximum number of conformer generation attempts
        allowed_atoms (set): Set of allowed atomic numbers. If provided, molecules with 
                           other atoms will be filtered out. Default: None (no filtering)
        
    Returns:
        tuple: (positions, atomic_numbers, num_atoms, processed_smiles) or None if failed
    """
    try:
        import warnings
        # Suppress NumPy/RDKit compatibility warnings
        warnings.filterwarnings('ignore', message='.*NumPy.*')
        warnings.filterwarnings('ignore', message='.*_ARRAY_API.*')
        
        import os
        # Set environment variable to suppress RDKit warnings
        os.environ['RDKIT_QUIET'] = '1'
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except (ImportError, AttributeError) as e:
        if "_ARRAY_API" in str(e):
            raise ImportError(f"RDKit/NumPy compatibility issue: {e}. Try: pip install --upgrade rdkit")
        else:
            raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        
        if num_atoms > 100:  # Skip very large molecules
            return None
            
        # Filter by allowed atom types BEFORE 3D optimization to save computation
        if allowed_atoms is not None:
            atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            unique_atoms = set(atomic_numbers)
            if not unique_atoms.issubset(allowed_atoms):
                # Molecule contains atoms not in allowed list - reject early
                return None
        
        # Filter out molecules with formal charges BEFORE 3D optimization to avoid UFFTYPER warnings
        has_formal_charges = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
        if has_formal_charges:
            # Reject molecules with formal charges (charged atoms cause UFFTYPER warnings)
            return None
        
        # Filter out molecules with complex sulfur atoms that cause UFFTYPER warnings
        # Both S_5+4 and S_6+6 warnings are problematic and hard to handle reliably
        has_complex_sulfur = any(
            atom.GetAtomicNum() == 16 and (atom.GetTotalValence() >= 6 or atom.GetDegree() >= 4)
            for atom in mol.GetAtoms()
        )
        if has_complex_sulfur:
            # Reject molecules with complex sulfur (SO2, SO3 groups cause various UFFTYPER warnings)
            # MMFF94 often fails to converge for these, and UFF produces warnings
            return None
        
        success = False
        for attempt in range(max_attempts):
            try:
                # Use basic embedding (fast) but ensure no UFF usage
                params = AllChem.EmbedParameters()
                params.randomSeed = 42 + attempt
                params.maxIterations = 200
                params.numThreads = 1
                
                result = AllChem.EmbedMolecule(mol, params)
                if result == 0:  # Success
                    # Use UFF for all remaining molecules (fast, no warnings since we filtered problematic ones)
                    try:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=200)  # Primary: UFF (fast)
                        success = True
                        break
                    except:
                        try:
                            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)  # Fallback: MMFF94
                            success = True
                            break
                        except:
                            continue
            except:
                continue
                
        if not success:
            return None
            
        positions = mol.GetConformer().GetPositions()
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        
        # Get the 3D SMILES after 3D generation and optimization (for validation)
        # Create a copy without explicit hydrogens for clean SMILES comparison
        mol_no_h = Chem.RemoveHs(mol)
        
        # Remove any radical electrons and generate clean canonical SMILES
        for atom in mol_no_h.GetAtoms():
            atom.SetNumRadicalElectrons(0)
        
        processed_smiles = Chem.MolToSmiles(mol_no_h, canonical=True, isomericSmiles=False)
        
        return positions, atomic_numbers, num_atoms, processed_smiles
    except Exception as e:
        return None

def process_smiles_to_npz(smiles_file, output_npz, max_molecules=None, test_mode=False, num_cores=None, allowed_atoms=None):
    """
    Process SMILES file to NPZ format with 3D coordinates.
    
    Args:
        smiles_file (str): Path to input SMILES file (.smiles or .csv)
        output_npz (str): Path to output NPZ file
        max_molecules (int): Maximum number of molecules to process
        test_mode (bool): If True, process only 100 molecules for testing
        num_cores (int): Number of CPU cores to use. If None, uses min(cpu_count(), 8)
        allowed_atoms (set): Set of allowed atomic numbers. If None, uses ALLOWED_ATOMS. 
                           Molecules with other atoms will be filtered out. {H,C,N,O,F,S,Cl,Br}
        
    Returns:
        dict: Processing statistics
    """
    print(f"Processing: {smiles_file} -> {output_npz}")
    
    # Use default allowed atoms if none specified
    if allowed_atoms is None:
        allowed_atoms = ALLOWED_ATOMS
        print(f"🎯 Filtering to allowed atoms: H, C, N, O, F, S, Cl, Br")
    
    # Read SMILES
    smiles_list = read_smiles_file(smiles_file)
    
    if test_mode:
        max_molecules = 100
        print(f"🧪 Test mode: Processing only {max_molecules} molecules")
    
    if max_molecules and max_molecules < len(smiles_list):
        smiles_list = smiles_list[:max_molecules]
        print(f"📏 Limited to {max_molecules} molecules")
    
    print(f"📊 Processing {len(smiles_list)} molecules...")
    
    # Process molecules using multiprocessing
    if num_cores is None:
        num_cores = min(cpu_count(), 8)  # Default: limit to 8 cores to avoid overwhelming system
    else:
        num_cores = min(num_cores, cpu_count())  # Ensure we don't exceed available cores
    
    print(f"🚀 Using {num_cores} CPU cores for parallel processing")
    
    all_positions = []
    all_charges = []
    all_num_atoms = []
    all_original_smiles = []  # Store original SMILES
    all_processed_smiles = []  # Store processed SMILES after 3D generation
    failed_count = 0
    
    # Process in parallel
    with Pool(processes=num_cores) as pool:
        # Use partial to pass allowed_atoms parameter
        process_func = partial(process_single_smiles, allowed_atoms=allowed_atoms)
        results = list(tqdm(pool.imap(process_func, smiles_list), 
                           total=len(smiles_list), desc="Converting SMILES to 3D"))
    
    # Collect successful results
    for i, result in enumerate(results):
        if result is not None:
            positions, charges, num_atoms, processed_smiles = result
            all_positions.append(positions)
            all_charges.append(charges)
            all_num_atoms.append(num_atoms)
            all_original_smiles.append(smiles_list[i])  # Store corresponding original SMILES
            all_processed_smiles.append(processed_smiles)  # Store processed SMILES
        else:
            failed_count += 1
    
    if not all_positions:
        raise ValueError("No molecules successfully processed!")
    
    success_count = len(all_positions)
    success_rate = success_count / len(smiles_list) * 100
    
    print(f"✅ Successfully processed: {success_count}/{len(smiles_list)} ({success_rate:.1f}%)")
    print(f"❌ Failed: {failed_count}")
    
    # Find maximum number of atoms for padding
    max_atoms = max(len(pos) for pos in all_positions)
    print(f"🔢 Max atoms per molecule: {max_atoms}")
    
    # Pad to uniform size
    padded_positions = np.zeros((success_count, max_atoms, 3), dtype=np.float32)
    padded_charges = np.zeros((success_count, max_atoms), dtype=np.int64)
    num_atoms_array = np.array(all_num_atoms, dtype=np.int64)
    
    for i, (pos, charges, n_atoms) in enumerate(zip(all_positions, all_charges, all_num_atoms)):
        padded_positions[i, :n_atoms, :] = pos
        padded_charges[i, :n_atoms] = charges
    
    # Create output directory
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    
    # Save NPZ file
    np.savez_compressed(
        output_npz,
        positions=padded_positions,
        charges=padded_charges,
        num_atoms=num_atoms_array,
        original_smiles=np.array(all_original_smiles, dtype=object),  # Store original SMILES
        processed_smiles=np.array(all_processed_smiles, dtype=object),  # Store processed SMILES
        # Metadata
        max_atoms=max_atoms,
        total_molecules=success_count,
        source_file=smiles_file,
        success_rate=success_rate
    )
    
    avg_atoms = np.mean(num_atoms_array)
    std_atoms = np.std(num_atoms_array)
    
    print(f"💾 Saved: {output_npz}")
    print(f"📏 Shape: {padded_positions.shape}")
    print(f"⚖️  Average atoms: {avg_atoms:.1f} ± {std_atoms:.1f}")
    
    return {
        'success_count': success_count,
        'failed_count': failed_count,
        'success_rate': success_rate,
        'max_atoms': max_atoms,
        'avg_atoms': avg_atoms,
        'output_shape': padded_positions.shape
    }

def read_smiles_file(file_path):
    """
    Read SMILES from file (.smiles or .csv).
    
    Args:
        file_path (str): Path to SMILES file
        
    Returns:
        list: List of SMILES strings
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.smiles':
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    elif file_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Try common SMILES column names
        smiles_columns = ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']
        smiles_col = None
        
        for col in smiles_columns:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            # Use first column if no standard name found
            smiles_col = df.columns[0]
            print(f"⚠️  Using column '{smiles_col}' as SMILES")
        
        return df[smiles_col].dropna().tolist()
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def remove_hydrogens_from_npz(input_npz, output_npz):
    """
    Remove hydrogen atoms from NPZ dataset.
    
    Args:
        input_npz (str): Path to input NPZ file (with hydrogens)
        output_npz (str): Path to output NPZ file (without hydrogens)
        
    Returns:
        dict: Processing statistics
    """
    print(f"Removing hydrogens: {input_npz} -> {output_npz}")
    
    # Load data
    data = np.load(input_npz, allow_pickle=True)
    positions = data['positions']
    charges = data['charges']
    num_atoms = data['num_atoms']
    original_smiles = data.get('original_smiles', None)  # Get original SMILES if available
    processed_smiles = data.get('processed_smiles', None)  # Get processed SMILES if available
    
    print(f"  📊 Original molecules: {len(num_atoms)}")
    
    # Process each molecule to remove hydrogens (Z=1)
    new_positions = []
    new_charges = []
    new_num_atoms = []
    new_original_smiles = []
    new_processed_smiles = []
    
    for mol_idx in tqdm(range(len(num_atoms)), desc="  Removing hydrogens"):
        n_atoms = num_atoms[mol_idx]
        mol_positions = positions[mol_idx][:n_atoms]
        mol_charges = charges[mol_idx][:n_atoms]
        
        # Find non-hydrogen atoms (atomic number != 1)
        non_h_mask = mol_charges != 1
        
        if np.any(non_h_mask):  # Make sure we have non-hydrogen atoms
            filtered_positions = mol_positions[non_h_mask]
            filtered_charges = mol_charges[non_h_mask]
            new_n_atoms = len(filtered_positions)
            
            new_positions.append(filtered_positions)
            new_charges.append(filtered_charges)
            new_num_atoms.append(new_n_atoms)
            if original_smiles is not None:
                new_original_smiles.append(original_smiles[mol_idx])
            if processed_smiles is not None:
                # Remove hydrogens from 3D SMILES to match the coordinates
                proc_smiles_no_h = remove_hydrogens_from_smiles(processed_smiles[mol_idx])
                if proc_smiles_no_h is not None:
                    new_processed_smiles.append(proc_smiles_no_h)
                else:
                    # Fallback to original if hydrogen removal fails
                    new_processed_smiles.append(processed_smiles[mol_idx])
        else:
            # Skip molecules with only hydrogens (shouldn't happen in practice)
            print(f"    ⚠️  Molecule {mol_idx} has only hydrogen atoms, skipping")
    
    if not new_positions:
        raise ValueError("No valid molecules after hydrogen removal!")
    
    print(f"  ✅ Molecules after filtering: {len(new_positions)}")
    
    # Find new max_atoms after hydrogen removal
    new_max_atoms = max(len(pos) for pos in new_positions)
    print(f"  🔢 Max atoms after H removal: {new_max_atoms}")
    
    # Pad to uniform size
    padded_positions = np.zeros((len(new_positions), new_max_atoms, 3), dtype=np.float32)
    padded_charges = np.zeros((len(new_positions), new_max_atoms), dtype=np.int64)
    new_num_atoms_array = np.array(new_num_atoms, dtype=np.int64)
    
    for i, (pos, charges_mol, n_atoms) in enumerate(zip(new_positions, new_charges, new_num_atoms)):
        padded_positions[i, :n_atoms, :] = pos
        padded_charges[i, :n_atoms] = charges_mol
    
    # Create output directory
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    
    # Save without hydrogen atoms
    save_data = {
        'positions': padded_positions,
        'charges': padded_charges,
        'num_atoms': new_num_atoms_array,
        # Add metadata
        'max_atoms': new_max_atoms,
        'total_molecules': len(new_positions),
        'source_file': input_npz,
        'hydrogen_removed': True
    }
    
    # Include original SMILES if available
    if original_smiles is not None and new_original_smiles:
        save_data['original_smiles'] = np.array(new_original_smiles, dtype=object)
    
    # Include processed SMILES if available
    if processed_smiles is not None and new_processed_smiles:
        save_data['processed_smiles'] = np.array(new_processed_smiles, dtype=object)
    
    np.savez_compressed(output_npz, **save_data)
    
    avg_atoms = np.mean(new_num_atoms_array)
    std_atoms = np.std(new_num_atoms_array)
    
    print(f"  💾 Saved: {len(new_positions)} molecules")
    print(f"  ⚖️  Average atoms per molecule: {avg_atoms:.1f} ± {std_atoms:.1f}")
    print(f"  📏 Output shape: {padded_positions.shape}")
    
    return {
        'molecules_processed': len(new_positions),
        'avg_atoms': avg_atoms,
        'max_atoms': new_max_atoms,
        'output_shape': padded_positions.shape
    }

def remove_hydrogens_from_smiles(smiles):
    """
    Remove explicit hydrogens from SMILES string using RDKit.
    
    Args:
        smiles (str): SMILES string with explicit hydrogens
        
    Returns:
        str: SMILES string without explicit hydrogens, or None if failed
    """
    try:
        from rdkit import Chem
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Remove hydrogens
        mol_no_h = Chem.RemoveHs(mol)
        
        # Convert back to SMILES
        smiles_no_h = Chem.MolToSmiles(mol_no_h, canonical=True)
        return smiles_no_h
        
    except Exception:
        return None

def get_smiles_from_3d_structure(positions, atomic_numbers):
    """
    Convert 3D structure back to SMILES string using RDKit.
    
    Args:
        positions (numpy.ndarray): 3D positions of atoms
        atomic_numbers (numpy.ndarray): Atomic numbers
        
    Returns:
        str: SMILES string or None if failed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Create RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for z in atomic_numbers:
            atom = Chem.Atom(int(z))
            mol.AddAtom(atom)
        
        # Set conformer
        conf = Chem.Conformer(len(atomic_numbers))
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
        mol.AddConformer(conf)
        
        # Convert to Mol object for bond perception
        mol = mol.GetMol()
        
        # Determine bonds based on distances
        Chem.rdDetermineBonds.DetermineBonds(mol, charge=0)
        
        # Sanitize the molecule
        Chem.SanitizeMol(mol)
        
        # Get canonical SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles
    except Exception as e:
        # If bond perception fails, try a simpler approach
        try:
            from rdkit import Chem
            mol = Chem.RWMol()
            
            # Add atoms
            for z in atomic_numbers:
                atom = Chem.Atom(int(z))
                mol.AddAtom(atom)
            
            # Set conformer
            conf = Chem.Conformer(len(atomic_numbers))
            for i, pos in enumerate(positions):
                conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
            mol.AddConformer(conf)
            
            mol = mol.GetMol()
            
            # Try basic distance-based bond assignment
            from rdkit.Chem import rdMolOps
            rdMolOps.ConnectTheDots(mol)
            
            # Try to sanitize
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles
            
        except Exception:
            return None

def create_mol2_samples(npz_file, output_dir, num_samples=10):
    """
    Convert NPZ data to MOL2 sample files using OpenBabel.
    
    Args:
        npz_file (str): Path to NPZ file
        output_dir (str): Output directory for MOL2 files
        num_samples (int): Number of sample molecules to create
        
    Returns:
        list: Paths to created MOL2 files
    """
    try:
        import openbabel
        import pybel
    except ImportError:
        print("❌ OpenBabel/pybel not available!")
        print("Install with: conda install -c conda-forge openbabel")
        return []
    
    print(f"Creating MOL2 samples from {npz_file}...")
    
    # Load NPZ data
    data = np.load(npz_file, allow_pickle=True)
    positions = data['positions']
    charges = data['charges']  # This is atomic numbers (Z)
    num_atoms = data['num_atoms']
    original_smiles = data.get('original_smiles', None)  # Get original SMILES if available
    processed_smiles = data.get('processed_smiles', None)  # Get processed SMILES if available
    
    # Select random samples
    num_molecules = len(num_atoms)
    if num_samples > num_molecules:
        num_samples = num_molecules
    
    np.random.seed(42)  # Reproducible sampling
    sample_indices = np.random.choice(num_molecules, size=num_samples, replace=False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dataset name from npz_file path
    dataset_name = Path(npz_file).stem  # train, valid, or test
    
    created_files = []
    for i, sample_idx in enumerate(sample_indices):
        n_atoms = num_atoms[sample_idx]
        mol_positions = positions[sample_idx][:n_atoms]
        mol_charges = charges[sample_idx][:n_atoms]
        
        # Get original SMILES for this molecule
        orig_smiles = original_smiles[sample_idx] if original_smiles is not None else "N/A"
        
        # We'll get the 3D SMILES from the actual MOL2 structure after creating it
        
        # Create OpenBabel molecule
        obmol = openbabel.OBMol()
        obmol.BeginModify()
        
        # Add atoms
        for j, (pos, z) in enumerate(zip(mol_positions, mol_charges)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(int(z))
            atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
        
        obmol.EndModify()
        
        # Perceive bonds and assign bond orders
        obmol.ConnectTheDots()  # Perceive bonds based on distances
        obmol.PerceiveBondOrders()  # Assign bond orders and aromaticity
        
        # Get 3D SMILES from OpenBabel reconstruction and canonicalize both for comparison
        obConversion_smiles = openbabel.OBConversion()
        obConversion_smiles.SetOutFormat("smi")
        mol2_smiles_raw = obConversion_smiles.WriteString(obmol).strip()
        
        if not mol2_smiles_raw:
            print(f"⚠️  Skipping molecule {sample_idx}: Could not generate SMILES from 3D coordinates")
            continue
        
        # Canonicalize both original and 3D-reconstructed SMILES using the same method
        try:
            from rdkit import Chem
            
            # Canonicalize original SMILES
            orig_mol = Chem.MolFromSmiles(orig_smiles)
            if orig_mol is None:
                print(f"⚠️  Skipping molecule {sample_idx}: Invalid original SMILES")
                continue
            
            # Remove hydrogens and radicals from original
            orig_mol_no_h = Chem.RemoveHs(orig_mol)
            for atom in orig_mol_no_h.GetAtoms():
                atom.SetNumRadicalElectrons(0)
            canonical_original = Chem.MolToSmiles(orig_mol_no_h, canonical=True, isomericSmiles=False)
            
            # Canonicalize 3D-reconstructed SMILES  
            mol2_mol = Chem.MolFromSmiles(mol2_smiles_raw)
            if mol2_mol is None:
                print(f"⚠️  Skipping molecule {sample_idx}: Invalid 3D-reconstructed SMILES")
                continue
            
            # Remove hydrogens and radicals from 3D reconstruction
            mol2_mol_no_h = Chem.RemoveHs(mol2_mol)
            for atom in mol2_mol_no_h.GetAtoms():
                atom.SetNumRadicalElectrons(0)
            canonical_3d = Chem.MolToSmiles(mol2_mol_no_h, canonical=True, isomericSmiles=False)
            
            # Check if canonical forms match
            if canonical_original != canonical_3d:
                print(f"⚠️  Skipping molecule {sample_idx}: 3D reconstruction doesn't match original")
                print(f"     Original canonical:  {canonical_original}")
                print(f"     3D canonical:        {canonical_3d}")
                continue
            
            # Use canonical form for both SMILES in the MOL2 file
            mol2_smiles = canonical_3d  # Same as canonical_original since they match
            
        except Exception as e:
            print(f"⚠️  Skipping molecule {sample_idx}: SMILES canonicalization failed - {e}")
            continue
        
        # Create title with SMILES information (using canonical forms)
        title = f"Molecule {sample_idx} from {dataset_name} dataset ({n_atoms} atoms)"
        if orig_smiles != "N/A":
            title += f" | Original SMILES: {canonical_original}"
        if mol2_smiles != "N/A":
            title += f" | 3D SMILES: {mol2_smiles}"
        
        obmol.SetTitle(title)
        
        # Write MOL2 file using OpenBabel
        mol2_filename = f"{dataset_name}_molecule_{i+1:02d}.mol2"
        mol2_path = os.path.join(output_dir, mol2_filename)
        
        obConversion = openbabel.OBConversion()
        obConversion.SetOutFormat("mol2")
        
        success = obConversion.WriteFile(obmol, mol2_path)
        if success:
            # Add SMILES as comments to the file
            with open(mol2_path, 'r') as f:
                content = f.read()
            
            # Insert SMILES comments after the molecule title
            lines = content.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                new_lines.append(line)
                if line.strip() == "@<TRIPOS>MOLECULE" and i + 1 < len(lines):
                    # Add the title line next
                    i += 1
                    new_lines.append(lines[i])
                    # Add SMILES information as comments (using canonical forms)
                    new_lines.append(f"# Original SMILES: {canonical_original}")
                    new_lines.append(f"# 3D SMILES:        {mol2_smiles}")
                i += 1
            
            # Write back the modified content
            with open(mol2_path, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print(f"  ✅ Created: {mol2_filename} ({n_atoms} atoms)")
            created_files.append(mol2_path)
        else:
            print(f"  ❌ Failed: {mol2_filename}")
    
    return created_files

def check_dependencies():
    """
    Check if required dependencies are available.
    
    Returns:
        dict: Status of each dependency
    """
    status = {}
    
    # Check RDKit
    try:
        import rdkit
        status['rdkit'] = True
    except ImportError:
        status['rdkit'] = False
    
    # Check OpenBabel
    try:
        import openbabel
        import pybel
        status['openbabel'] = True
    except ImportError:
        status['openbabel'] = False
    
    # Check pandas
    try:
        import pandas
        status['pandas'] = True
    except ImportError:
        status['pandas'] = False
    
    return status

def print_dependency_status():
    """Print status of required dependencies."""
    status = check_dependencies()
    
    print("🔍 Dependency Check:")
    print(f"  RDKit:     {'✅' if status['rdkit'] else '❌'}")
    print(f"  OpenBabel: {'✅' if status['openbabel'] else '❌'}")
    print(f"  Pandas:    {'✅' if status['pandas'] else '❌'}")
    
    if not all(status.values()):
        print("\n📦 Install missing dependencies:")
        if not status['rdkit']:
            print("  conda install -c conda-forge rdkit")
        if not status['openbabel']:
            print("  conda install -c conda-forge openbabel")
        if not status['pandas']:
            print("  conda install pandas")
    
    return all(status.values())
