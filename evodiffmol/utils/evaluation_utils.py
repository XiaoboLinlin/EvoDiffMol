#!/usr/bin/env python3

import os
import logging
import builtins
from rdkit import Chem
from utils.reconstruct import build_molecule
from utils.mol2_writer import write_mol2_structure


def setup_comprehensive_logging(args, remove_h):
    """
    Set up comprehensive logging to both console and file.
    
    Args:
        args: Command line arguments containing output_dir, dataset, num_samples
        remove_h: Whether hydrogens are removed
        
    Returns:
        tuple: (logger, original_print_function, log_path)
    """
    # Set up logger
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up log filename based on dataset and configuration
    dataset_suffix = f"{args.dataset}_{'without_h' if remove_h else 'with_h'}"
    log_filename = f"generation_log_{dataset_suffix}_{args.num_samples}.log"
    log_path = os.path.join(args.output_dir, log_filename)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(name)s::%(levelname)s] %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"📝 Logging to: {log_path}")
    logger.info(args)
    
    # Set up a custom print function that also logs to file
    original_print = print
    def custom_print(*args, **kwargs):
        # Print to console
        original_print(*args, **kwargs)
        # Also log to file
        message = ' '.join(str(arg) for arg in args)
        if message.strip():  # Only log non-empty messages
            logger.info(f"[PRINT] {message}")
    
    # Replace print function globally for molecule building functions
    builtins.print = custom_print
    
    return logger, original_print, log_path


def save_generated_smiles(smile_list, args, remove_h, logger):
    """
    Save valid SMILES to files for MOSES evaluation.
    
    Args:
        smile_list: List of SMILES strings (may contain None values)
        args: Command line arguments containing output_dir, dataset, num_samples
        remove_h: Whether hydrogens are removed
        logger: Logger instance
        
    Returns:
        tuple: (valid_smiles_count, unique_smiles_count, smiles_path, unique_path)
    """
    if not smile_list:
        logger.warning("❌ No valid SMILES generated - no output file created")
        return 0, 0, None, None
    
    # Filter out None values to get only valid SMILES
    valid_smiles = [smile for smile in smile_list if smile is not None]
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine output filename based on dataset and configuration
    dataset_suffix = f"{args.dataset}_{'without_h' if remove_h else 'with_h'}"
    smiles_filename = f"generated_smiles_{dataset_suffix}_{args.num_samples}.txt"
    smiles_path = os.path.join(output_dir, smiles_filename)
    
    # Save SMILES - one per line
    with open(smiles_path, 'w') as f:
        for smile in valid_smiles:
            f.write(f"{smile}\n")
    
    logger.info(f"✅ Saved {len(valid_smiles)} valid SMILES to: {smiles_path}")
    
    # Also save unique SMILES separately
    unique_smiles = list(set(valid_smiles))
    unique_filename = f"unique_smiles_{dataset_suffix}_{args.num_samples}.txt"
    unique_path = os.path.join(output_dir, unique_filename)
    
    with open(unique_path, 'w') as f:
        for smile in unique_smiles:
            f.write(f"{smile}\n")
    
    logger.info(f"✅ Saved {len(unique_smiles)} unique SMILES to: {unique_path}")
    
    # Print summary for MOSES evaluation
    logger.info("=" * 50)
    logger.info("SMILES Generation Summary:")
    logger.info(f"Total samples requested: {args.num_samples}")
    logger.info(f"Valid SMILES generated: {len(valid_smiles)}")
    logger.info(f"Unique SMILES: {len(unique_smiles)}")
    logger.info(f"Validity rate: {len(valid_smiles)/args.num_samples:.3f}")
    logger.info(f"Uniqueness rate: {len(unique_smiles)/len(valid_smiles):.3f}" if valid_smiles else "Uniqueness rate: 0.000")
    logger.info("=" * 50)
    
    return len(valid_smiles), len(unique_smiles), smiles_path, unique_path


def process_molecule_for_mol2(pos, atom_type, smile, i, batch_size, dataset_info, args, remove_h, logger, update_mask=None):
    """
    Process a single molecule for MOL2 writing and kekulization testing.

    Args:
        pos: Position tensor
        atom_type: Atom type tensor
        smile: SMILES string
        i: Molecule index
        batch_size: Batch size
        dataset_info: Dataset configuration
        args: Command line arguments
        remove_h: Whether hydrogens are removed
        logger: Logger instance
        update_mask: Optional tensor indicating fixed atoms (0=fixed, 1=generated)

    Returns:
        dict: Results dictionary with molecule data and status
    """
    # Create dataset-specific directory name
    dataset_name = args.dataset
    if remove_h:
        dataset_name += '_without_h'
    output_dir = getattr(args, 'output_dir', 'results')
    sdf_dir = f'{output_dir}/{dataset_name}'
    if not os.path.exists(sdf_dir):
        os.makedirs(sdf_dir)
    base_path = os.path.join(sdf_dir, f'full_{i//batch_size}_{i%batch_size}')
    
    # Rebuild molecule for MOL2 writing
    mol = build_molecule(pos, atom_type, dataset_info)
    
    # Get SMILES from 3D structure (skip kekulization test)
    try:
        mol3d_smiles = Chem.MolToSmiles(mol)
    except:
        mol3d_smiles = None
    
    # Write MOL2 file with update_mask information for fixed structure visualization
    mol2_path = base_path + ".mol2"
    success = write_mol2_structure(mol, pos, smile, mol2_path, update_mask=update_mask)
    
    if not success:
        logger.warning(f"MOL2 structure writing failed for molecule {i//batch_size}_{i%batch_size}")
    
    # Return results
    return {
        'atom_type': atom_type,
        'pos': pos,
        'smile': smile,
        'mol2_success': success,
        'mol3d_smiles': mol3d_smiles
    }


def convert_to_standard_smiles(square_bracket_smiles):
    """
    Convert square bracket SMILES notation to standard SMILES notation.
    
    Args:
        square_bracket_smiles: SMILES with square brackets like [C][C](c1[c][c][c]c...)
        
    Returns:
        str: Standard SMILES notation, or original if conversion fails
    """
    try:
        # Parse the square bracket SMILES
        mol = Chem.MolFromSmiles(square_bracket_smiles)
        if mol is not None:
            # Generate standard SMILES
            standard_smiles = Chem.MolToSmiles(mol, canonical=True)
            return standard_smiles
        else:
            return square_bracket_smiles
    except:
        return square_bracket_smiles


def test_3d_structure_kekulization(mol, i, batch_size, logger):
    """
    Test if the 3D structure can be kekulized by converting it to SMILES first.
    
    Args:
        mol: RDKit molecule object
        i: Molecule index
        batch_size: Batch size
        logger: Logger instance
    
    Returns:
        tuple: (kekulizable, mol3d_smiles)
    """
    try:
        # Convert the 3D molecule to SMILES
        mol3d_smiles = Chem.MolToSmiles(mol)
        if mol3d_smiles is None:
            logger.warning(f"Failed to convert 3D structure to SMILES for molecule {i//batch_size}_{i%batch_size}")
            return False, None
        
        # Test if the 3D structure SMILES can be kekulized
        test_mol = Chem.MolFromSmiles(mol3d_smiles)
        if test_mol is None:
            logger.warning(f"3D structure SMILES is invalid for molecule {i//batch_size}_{i%batch_size}: {mol3d_smiles}")
            return False, mol3d_smiles
        else:
            Chem.Kekulize(test_mol)
            return True, mol3d_smiles
            
    except Exception as e:
        # Log the actual error for debugging
        logger.warning(f"3D structure kekulization failed for molecule {i//batch_size}_{i%batch_size}: {e}")
        return False, None


def save_evaluation_results(results_list, args, remove_h, logger):
    """
    Save evaluation results to log file and pickle files.

    Args:
        results_list: List of result dictionaries
        args: Command line arguments
        remove_h: Whether hydrogens are removed
        logger: Logger instance
    """
    # Create results directory
    dataset_name = args.dataset
    if remove_h:
        dataset_name += '_without_h'
    output_dir = getattr(args, 'output_dir', 'results')
    results_dir = f'{output_dir}/{dataset_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save evaluation log
    eval_log_path = os.path.join(results_dir, 'evaluation_results.log')
    with open(eval_log_path, 'w') as f:
        f.write(f"=== {args.dataset.upper()} Evaluation Results ===\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Remove H: {remove_h}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Sampling method: {args.sampling_type}\n")
        f.write(f"Weights - Global pos: {args.w_global_pos}, Global node: {args.w_global_node}, Local pos: {args.w_local_pos}, Local node: {args.w_local_node}\n")
        f.write(f"\n=== Results ===\n")
        f.write(f"Total molecules: {len(results_list)}\n")
    
    print(f"Evaluation results saved to: {eval_log_path}", flush=True)
    
    # Save pickle files for large datasets
    if args.num_samples >= 10000:
        save_path = os.path.join(results_dir, 'samples_all.pkl')
        save_smile_path = os.path.join(results_dir, 'samples_smile.pkl')
        
        logger.info('Saving samples to: %s' % save_path)
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(results_list, f)
        
        # Extract SMILES for separate file
        smile_list = [r['smile'] for r in results_list if r.get('smile') is not None]
        with open(save_smile_path, 'wb') as f:
            pickle.dump(smile_list, f)


def log_generation_statistics(position_list, atom_type_list, smile_list, valid, stable, logger, context="Generation", requested_samples=None, total_generated_molecules=None):
    """
    Log generation statistics in a consistent format.
    
    Args:
        position_list: List of position tensors (only contains valid molecules)
        atom_type_list: List of atom type tensors
        smile_list: List of SMILES strings
        valid: Number of valid molecules
        stable: Number of stable molecules
        logger: Logger instance
        context: Context string for logging (e.g., "Generation", "Evaluation")
        requested_samples: Total number of samples requested
        total_generated_molecules: Total molecules generated by diffusion (before validation)
    """
    kept_molecules = len(position_list)  # This equals 'valid' by design
    
    logger.info(f"{context} statistics:")
    if kept_molecules > 0:
        # Assert the logical consistency
        assert kept_molecules == valid, f"Logic error: kept_molecules ({kept_molecules}) should equal valid ({valid})"
        
        # Use total_generated_molecules as denominator if available, otherwise fall back to requested_samples
        denominator = total_generated_molecules if total_generated_molecules is not None else requested_samples
        
        if denominator is not None and denominator != kept_molecules:
            logger.info(f"  - Generated: {kept_molecules}/{denominator} ({kept_molecules/denominator:.4f}) - {denominator - kept_molecules} failed")
        logger.info(f"  - Valid: {valid} molecules")
        logger.info(f"  - Stable: {stable}/{valid} ({stable/valid:.4f})")
        logger.info(f"  - Unique: {len(set(smile_list))}/{len(smile_list)} ({len(set(smile_list))/len(smile_list) if len(smile_list) > 0 else 0:.4f})")
    else:
        denominator = total_generated_molecules if total_generated_molecules is not None else requested_samples
        if denominator is not None:
            logger.warning(f"  - Generated: 0/{denominator} (0.0000) - all {denominator} failed")
        else:
            logger.warning(f"  - No molecules generated - all batches failed during sampling")
        logger.info(f"  - Valid: 0/0 (0.0000)")
        logger.info(f"  - Stable: 0/0 (0.0000)")
        logger.info(f"  - Unique: 0/0 (0.0000)")


def save_trajectory_mol2_files(trajectory_data, timesteps, output_dir, dataset_info, logger):
    """
    Save trajectory data as animated MOL2 files with bond information.
    Creates only animated multi-frame files with bonds for the final molecule (t=1000).
    
    Args:
        trajectory_data: {molecule_idx: {timestep: (pos, atom_type)}}
        timesteps: List of timesteps
        output_dir: Base output directory
        dataset_info: Dataset information
        logger: Logger instance
    """
    from utils.reconstruct import build_molecule
    import torch
    
    trajectory_dir = os.path.join(output_dir, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    # Handle atom decoder
    atom_decoder = dataset_info.get('atom_decoder', [])
    if isinstance(atom_decoder, list):
        atom_decoder = {i: atom for i, atom in enumerate(atom_decoder)}
    
    # Create animated MOL2 files with bonds from build_molecule
    for mol_idx in trajectory_data:
        filename = f"molecule_{mol_idx:03d}_animated.mol2"
        filepath = os.path.join(trajectory_dir, filename)
        
        with open(filepath, 'w') as f:
            for frame_idx, timestep in enumerate(sorted(timesteps, reverse=True)):  # Start from highest timestep (t=1000)
                if timestep in trajectory_data[mol_idx]:
                    pos, atom_type, update_mask = trajectory_data[mol_idx][timestep]
                    
                    # Convert tensors if needed
                    if hasattr(pos, 'cpu'):
                        pos_tensor = pos.cpu()
                        pos_np = pos.cpu().numpy()
                    else:
                        pos_tensor = torch.tensor(pos)
                        pos_np = pos
                    
                    if hasattr(atom_type, 'cpu'):
                        atom_type_tensor = atom_type.cpu()
                        atom_type_np = atom_type.cpu().numpy()
                    else:
                        atom_type_tensor = torch.tensor(atom_type)
                        atom_type_np = atom_type
                    
                    # For t=0, save two frames: one without bonds, one with bonds
                    if timestep == 0:
                        # Frame 1: t=0 without bonds (raw atomic positions)
                        f.write("@<TRIPOS>MOLECULE\n")
                        f.write(f"diffusion_frame_{frame_idx:03d}_t{timestep}_atoms_only\n")
                        f.write(f"{len(pos_np)} 0 0 0 0\n")
                        f.write("SMALL\n")
                        f.write("GASTEIGER\n")
                        f.write("\n")
                        
                        # Write atoms with scaffold/generated distinction
                        f.write("@<TRIPOS>ATOM\n")
                        for i, (position, atom_type_idx) in enumerate(zip(pos_np, atom_type_np)):
                            atom_symbol = atom_decoder.get(int(atom_type_idx), f"X{atom_type_idx}")
                            
                            # Create descriptive atom name based on scaffold vs generated
                            if update_mask is not None:
                                if update_mask[i] == 0:  # Fixed scaffold atom
                                    atom_name = f"FIXED_{atom_symbol}{i+1}"
                                    subst_name = "SCAFFOLD"
                                else:  # Generated atom
                                    atom_name = f"GEN_{atom_symbol}{i+1}"
                                    subst_name = "GENERATED"
                            else:
                                atom_name = f"{atom_symbol}{i+1}"
                                subst_name = "MOL"
                            
                            f.write(f"{i+1:6d} {atom_name:<12s} {position[0]:10.4f} {position[1]:10.4f} {position[2]:10.4f} {atom_symbol:<6s} 1 {subst_name:<8s} 0.0000\n")
                        
                        # Write substructure
                        f.write("\n@<TRIPOS>SUBSTRUCTURE\n")
                        if update_mask is not None:
                            f.write("1 SCAFFOLD 1 TEMP 0 **** **** 0 ROOT\n")
                            f.write("2 GENERATED 1 TEMP 0 **** **** 0 ROOT\n")
                        else:
                            f.write("1 MOL 1 TEMP 0 **** **** 0 ROOT\n")
                        f.write("\n")
                        
                        # Frame 2: t=0 with bonds (built molecule)
                        bonds = []
                        try:
                            mol = build_molecule(pos_tensor, atom_type_tensor, dataset_info, method='openbabel')
                            if mol is not None:
                                # Extract bonds from RDKit molecule
                                for bond in mol.GetBonds():
                                    atom1 = bond.GetBeginAtomIdx() + 1  # 1-indexed for MOL2
                                    atom2 = bond.GetEndAtomIdx() + 1
                                    bond_order = int(bond.GetBondType())
                                    bonds.append((atom1, atom2, bond_order))
                        except Exception as e:
                            logger.warning(f"Failed to build molecule for mol {mol_idx} timestep {timestep}: {e}")
                        
                        f.write("@<TRIPOS>MOLECULE\n")
                        f.write(f"diffusion_frame_{frame_idx:03d}_t{timestep}_with_bonds\n")
                        f.write(f"{len(pos_np)} {len(bonds)} 0 0 0\n")
                        f.write("SMALL\n")
                        f.write("GASTEIGER\n")
                        f.write("\n")
                        
                        # Write atoms with scaffold/generated distinction
                        f.write("@<TRIPOS>ATOM\n")
                        for i, (position, atom_type_idx) in enumerate(zip(pos_np, atom_type_np)):
                            atom_symbol = atom_decoder.get(int(atom_type_idx), f"X{atom_type_idx}")
                            
                            # Create descriptive atom name based on scaffold vs generated
                            if update_mask is not None:
                                if update_mask[i] == 0:  # Fixed scaffold atom
                                    atom_name = f"FIXED_{atom_symbol}{i+1}"
                                    subst_name = "SCAFFOLD"
                                else:  # Generated atom
                                    atom_name = f"GEN_{atom_symbol}{i+1}"
                                    subst_name = "GENERATED"
                            else:
                                atom_name = f"{atom_symbol}{i+1}"
                                subst_name = "MOL"
                            
                            f.write(f"{i+1:6d} {atom_name:<12s} {position[0]:10.4f} {position[1]:10.4f} {position[2]:10.4f} {atom_symbol:<6s} 1 {subst_name:<8s} 0.0000\n")
                        
                        # Write bonds if available
                        if bonds:
                            f.write("\n@<TRIPOS>BOND\n")
                            for bond_idx, (atom1, atom2, bond_order) in enumerate(bonds):
                                bond_type = "1"  # Default single
                                if bond_order == 2:
                                    bond_type = "2"
                                elif bond_order == 3:
                                    bond_type = "3"
                                elif bond_order == 12:  # Aromatic
                                    bond_type = "ar"
                                
                                f.write(f"{bond_idx+1:6d} {atom1:6d} {atom2:6d} {bond_type}\n")
                        
                        # Write substructure
                        f.write("\n@<TRIPOS>SUBSTRUCTURE\n")
                        if update_mask is not None:
                            f.write("1 SCAFFOLD 1 TEMP 0 **** **** 0 ROOT\n")
                            f.write("2 GENERATED 1 TEMP 0 **** **** 0 ROOT\n")
                        else:
                            f.write("1 MOL 1 TEMP 0 **** **** 0 ROOT\n")
                        f.write("\n")
                    
                    else:
                        # For other timesteps, just save atoms only
                        f.write("@<TRIPOS>MOLECULE\n")
                        f.write(f"diffusion_frame_{frame_idx:03d}_t{timestep}\n")
                        f.write(f"{len(pos_np)} 0 0 0 0\n")
                        f.write("SMALL\n")
                        f.write("GASTEIGER\n")
                        f.write("\n")
                        
                        # Write atoms with scaffold/generated distinction
                        f.write("@<TRIPOS>ATOM\n")
                        for i, (position, atom_type_idx) in enumerate(zip(pos_np, atom_type_np)):
                            atom_symbol = atom_decoder.get(int(atom_type_idx), f"X{atom_type_idx}")
                            
                            # Create descriptive atom name based on scaffold vs generated
                            if update_mask is not None:
                                if update_mask[i] == 0:  # Fixed scaffold atom
                                    atom_name = f"FIXED_{atom_symbol}{i+1}"
                                    subst_name = "SCAFFOLD"
                                else:  # Generated atom
                                    atom_name = f"GEN_{atom_symbol}{i+1}"
                                    subst_name = "GENERATED"
                            else:
                                atom_name = f"{atom_symbol}{i+1}"
                                subst_name = "MOL"
                            
                            f.write(f"{i+1:6d} {atom_name:<12s} {position[0]:10.4f} {position[1]:10.4f} {position[2]:10.4f} {atom_symbol:<6s} 1 {subst_name:<8s} 0.0000\n")
                        
                        # Write substructure
                        f.write("\n@<TRIPOS>SUBSTRUCTURE\n")
                        if update_mask is not None:
                            f.write("1 SCAFFOLD 1 TEMP 0 **** **** 0 ROOT\n")
                            f.write("2 GENERATED 1 TEMP 0 **** **** 0 ROOT\n")
                        else:
                            f.write("1 MOL 1 TEMP 0 **** **** 0 ROOT\n")
                        f.write("\n")
    
    # Create summary file
    summary_file = os.path.join(trajectory_dir, "README.txt")
    with open(summary_file, 'w') as f:
        f.write("Animated Diffusion Trajectory Visualization\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of molecules: {len(trajectory_data)}\n")
        f.write(f"Timesteps per molecule: {len(timesteps)}\n")
        f.write(f"Timesteps: {timesteps}\n\n")
        f.write("File format: Animated MOL2 with bonds\n")
        f.write("- Multi-frame MOL2 files (one per molecule)\n")
        f.write("- Frames ordered from final molecule (t=1000) to early diffusion (t=0)\n")
        f.write("- Bond information included for final molecule (t=1000)\n")
        f.write("- Earlier timesteps show atoms only (no bonds)\n\n")
        f.write("Directory structure:\n")
        f.write("trajectory/\n")
        f.write("├── molecule_000_animated.mol2  # Multi-frame animation\n")
        f.write("├── molecule_001_animated.mol2  # Multi-frame animation\n")
        f.write("└── README.txt\n\n")
        f.write("Visualization:\n")
        f.write("- PyMOL: load molecule_000_animated.mol2 (use 'play' button)\n")
        f.write("- ChimeraX: open molecule_000_animated.mol2 (Movie → Play)\n")
        f.write("- VMD: mol load mol2 molecule_000_animated.mol2\n")
        f.write("- MDAnalysis: u = mda.Universe('molecule_000_animated.mol2')\n\n")
        f.write("Animation sequence:\n")
        f.write("Frame 0: Final clean molecule (t=0) - ATOMS ONLY (raw positions)\n")
        f.write("Frame 1: Final clean molecule (t=0) - WITH BONDS (built structure)\n")
        f.write("Frame 2: Intermediate diffusion (t=500) - atoms only\n")
        f.write("Frame 3: Initial noisy state (t=1000) - atoms only\n\n")
        f.write("Note: t=0 has two frames to show the difference between:\n")
        f.write("- Raw atomic positions from diffusion model\n")
        f.write("- Final molecular structure with bonds\n\n")
        f.write("Scaffold-based generation:\n")
        f.write("- FIXED_* atoms: Scaffold atoms (constrained during diffusion)\n")
        f.write("- GEN_* atoms: Generated atoms (free to move during diffusion)\n")
        f.write("- SCAFFOLD substructure: Fixed scaffold atoms\n")
        f.write("- GENERATED substructure: Newly generated atoms\n")
    
    logger.info(f"🎬 Animated trajectory files saved to: {trajectory_dir}")
    logger.info(f"   {len(trajectory_data)} animated MOL2 files created")
    logger.info(f"   Each file: {len(timesteps)+1} frames (t=0 has 2 frames: atoms only + with bonds)")
    logger.info(f"   Final clean molecule (t=0) shows raw positions AND built structure")
    logger.info(f"   Load pattern: molecule_000_animated.mol2")





def trajectory_with_model(model, dataset_info, num_samples, batch_size, sampling_params, 
                         context=None, device='cuda', model_config=None, logger=None, return_trajectory=False):
    """
    Generate molecules using a trained model and return raw positions/atom_types before molecule building.
    
    Args:
        model: Trained model for molecule generation
        dataset_info: Dataset configuration
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        sampling_params: Dictionary of sampling parameters
        context: Optional context for conditional generation
        device: Device to run generation on
        model_config: Optional model configuration
        logger: Optional logger for output
        return_trajectory: If True, return full trajectory (pos_traj, atom_traj)
    
    Returns:
        If return_trajectory=False: (position_list, atom_type_list)
        If return_trajectory=True: (position_list, atom_type_list, pos_traj_list, atom_traj_list)
    """
    from utils.sample import construct_dataset
    from torch_geometric.data import Batch
    from torch_geometric.utils import unbatch
    import torch
    from tqdm import tqdm
    
    # Use passed logger or fall back to module logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Construct dataset for sampling
    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)
    
    position_list = []
    atom_type_list = []
    pos_traj_list = [] if return_trajectory else None
    atom_traj_list = [] if return_trajectory else None
    
    model.eval()
    
    with torch.no_grad():
        for n, datas in enumerate(tqdm(data_list, desc='Generating trajectory molecules')):
            batch = Batch.from_data_list(datas).to(device)
            
            try:
                # Generate molecules using model sampling
                pos_gen, pos_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                    atom_type=batch.x,
                    pos_init=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=None,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    context=context,
                    extend_order=False,
                    n_steps=sampling_params.get('n_steps', model_config.model.num_diffusion_timesteps if model_config else 1000),
                    step_lr=sampling_params.get('step_lr', 1e-6),
                    w_global_pos=sampling_params.get('w_global_pos', 1.0),
                    w_global_node=sampling_params.get('w_global_node', 4.0),
                    w_local_pos=sampling_params.get('w_local_pos', 1.0),
                    w_local_node=sampling_params.get('w_local_node', 5.0),
                    global_start_sigma=sampling_params.get('global_start_sigma', float('inf')),
                    clip=sampling_params.get('clip', 1000.0),
                    sampling_type=sampling_params.get('sampling_type', 'generalized'),
                    eta=sampling_params.get('eta', 1.0),
                )
                
                # Unbatch the generated molecules
                pos_list = unbatch(pos_gen, batch.batch)
                atom_type_list_batch = unbatch(atom_type, batch.batch)
                
                # Process each molecule in the batch
                for m in range(len(pos_list)):
                    pos = pos_list[m]
                    atom_type_final = atom_type_list_batch[m]

                    # Remove charge and get atom type indices for final result
                    atom_type_final = atom_type_final[:, :-1]
                    atom_type_final = torch.argmax(atom_type_final, dim=1)
                    
                    # Store final results
                    position_list.append(pos.cpu().detach())
                    atom_type_list.append(atom_type_final.cpu().detach())
                    
                    # Extract trajectory for this molecule if requested
                    if return_trajectory:
                        mol_pos_traj = []
                        mol_atom_traj = []
                        
                        for t_idx in range(len(pos_traj)):
                            pos_t_list = unbatch(pos_traj[t_idx], batch.batch, dim=0)
                            atom_t_list = unbatch(atom_traj[t_idx], batch.batch, dim=0)
                            
                            mol_pos_traj.append(pos_t_list[m].detach().cpu())
                            mol_atom_traj.append(atom_t_list[m].detach().cpu())
                        
                        pos_traj_list.append(mol_pos_traj)
                        atom_traj_list.append(mol_atom_traj)
                    
            except Exception as e:
                logger.warning(f"Failed to generate batch {n}: {e}")
                continue
    
    if return_trajectory:
        return position_list, atom_type_list, pos_traj_list, atom_traj_list
    else:
        return position_list, atom_type_list


def trajectory_with_fixed_structure(model, dataset_info, num_samples, batch_size, sampling_params, 
                                   mol2_path, context=None, device='cuda', model_config=None, 
                                   remove_h=True, positioning_strategy='plane_through_origin', logger=None, return_trajectory=False):
    """
    Generate molecules with fixed structure and return raw positions/atom_types before molecule building.
    
    Args:
        model: Trained diffusion model
        dataset_info: Dataset configuration
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        sampling_params: Dictionary of sampling parameters
        mol2_path: Path to MOL2 file containing fixed structure
        context: Optional context
        device: Device to run on
        model_config: Model configuration
        remove_h: Whether to remove hydrogens
        positioning_strategy: Scaffold positioning strategy
        logger: Optional logger
        return_trajectory: If True, return full trajectory (pos_traj, atom_traj)
        
    Returns:
        If return_trajectory=False: (position_list, atom_type_list, update_masks_list)
        If return_trajectory=True: (position_list, atom_type_list, update_masks_list, pos_traj_list, atom_traj_list)
    """
    from utils.mol2_parser import parse_mol2_fixed_structure
    from utils.sample import construct_dataset_with_fixed
    from torch_geometric.data import Batch
    from torch_geometric.utils import unbatch
    import torch
    from tqdm import tqdm
    
    # Use passed logger or fall back to module logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Parse fixed structure from MOL2
    fixed_structure = parse_mol2_fixed_structure(mol2_path, remove_h=remove_h, dataset_info=dataset_info)
    
    # Construct dataset with fixed structure
    data_list, total_samples, update_masks = construct_dataset_with_fixed(
        num_samples=num_samples, 
        batch_size=batch_size, 
        dataset_info=dataset_info, 
        fixed_structure=fixed_structure,
        positioning_strategy=positioning_strategy
    )
    
    model.eval()
    
    position_list = []
    atom_type_list = []
    update_masks_list = []
    pos_traj_list = [] if return_trajectory else None
    atom_traj_list = [] if return_trajectory else None
    
    with torch.no_grad():
        for batch_idx, (batch_data_list, batch_update_masks_list) in enumerate(zip(data_list, update_masks)):
            try:
                # Convert list of Data objects to Batch
                batch_data = Batch.from_data_list(batch_data_list).to(device)
                
                # Convert list of update masks to single tensor for batch
                if batch_update_masks_list is not None:
                    batch_update_masks = torch.cat(batch_update_masks_list, dim=0).to(device)
                else:
                    batch_update_masks = None
                
                current_batch_size = batch_data.num_graphs
                
                # Run diffusion sampling
                pos_gen, pos_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                    atom_type=batch_data.x,
                    pos_init=batch_data.pos,
                    bond_index=batch_data.edge_index,
                    bond_type=None,
                    batch=batch_data.batch,
                    num_graphs=current_batch_size,
                    context=context,
                    extend_order=False,
                    n_steps=sampling_params['n_steps'],
                    step_lr=sampling_params['step_lr'],
                    w_global_pos=sampling_params['w_global_pos'],
                    w_global_node=sampling_params['w_global_node'],
                    w_local_pos=sampling_params['w_local_pos'],
                    w_local_node=sampling_params['w_local_node'],
                    global_start_sigma=sampling_params['global_start_sigma'],
                    clip=sampling_params['clip'],
                    sampling_type=sampling_params['sampling_type'],
                    eta=sampling_params['eta'],
                    update_mask=batch_update_masks
                )
                
                # Unbatch results
                pos_list = unbatch(pos_gen, batch_data.batch, dim=0)
                atom_type_list_batch = unbatch(atom_type, batch_data.batch, dim=0)
                update_mask_list = unbatch(batch_update_masks, batch_data.batch, dim=0) if batch_update_masks is not None else [None] * current_batch_size
                
                # Process each molecule in the batch
                for m in range(current_batch_size):
                    pos = pos_list[m]
                    atom_type = atom_type_list_batch[m]
                    mol_update_mask = update_mask_list[m]

                    # Remove charge and get atom type indices
                    atom_type = atom_type[:, :-1]
                    atom_type = torch.argmax(atom_type, dim=1)
                    
                    # Store raw positions and atom types (before molecule building)
                    position_list.append(pos.detach().cpu())
                    atom_type_list.append(atom_type.detach().cpu())
                    update_masks_list.append(mol_update_mask.detach().cpu() if mol_update_mask is not None else None)
                    
                    # Extract trajectory for this molecule if requested
                    if return_trajectory:
                        mol_pos_traj = []
                        mol_atom_traj = []
                        
                        for t_idx in range(len(pos_traj)):
                            pos_t_list = unbatch(pos_traj[t_idx], batch_data.batch, dim=0)
                            atom_t_list = unbatch(atom_traj[t_idx], batch_data.batch, dim=0)
                            
                            mol_pos_traj.append(pos_t_list[m].detach().cpu())
                            mol_atom_traj.append(atom_t_list[m].detach().cpu())
                        
                        pos_traj_list.append(mol_pos_traj)
                        atom_traj_list.append(mol_atom_traj)
                    
            except Exception as e:
                logger.warning(f"Failed to generate batch {batch_idx}: {e}")
                continue
    
    if return_trajectory:
        return position_list, atom_type_list, update_masks_list, pos_traj_list, atom_traj_list
    else:
        return position_list, atom_type_list, update_masks_list



def generate_trajectory_for_timesteps(model, dataset_info, num_samples, batch_size, base_sampling_params, 
                                    timesteps, mol2_path=None, positioning_strategy='plane_through_origin', 
                                    device='cuda', model_config=None, logger=None):
    """
    Generate trajectory by running diffusion ONCE and extracting positions at specific timesteps.
    This ensures consistent atom counts across all timesteps for proper animation.
    
    Args:
        model: Diffusion model
        dataset_info: Dataset information
        num_samples: Number of molecules to generate
        batch_size: Batch size
        base_sampling_params: Base sampling parameters
        timesteps: List of timesteps to extract from trajectory
        mol2_path: Optional MOL2 file for scaffold mode
        positioning_strategy: Positioning strategy for scaffolds
        device: Device to use
        model_config: Model configuration
        logger: Logger instance
        
    Returns:
        dict: {molecule_idx: {timestep: (pos, atom_type, update_mask)}}
    """
    trajectory_data = {}
    
    if logger:
        logger.info(f"Generating full diffusion trajectory and extracting timesteps: {timesteps}")
    
    # Run full diffusion sampling once (n_steps = max timestep or 1000)
    max_timestep = max(timesteps) if timesteps else 1000
    full_sampling_params = base_sampling_params.copy()
    full_sampling_params['n_steps'] = max_timestep
    
    if mol2_path:
        # Fixed structure generation - run once and get full trajectory
        pos_list, atom_list, update_masks_list, pos_traj_list, atom_traj_list = trajectory_with_fixed_structure(
            model=model,
            dataset_info=dataset_info,
            num_samples=num_samples,
            batch_size=batch_size,
            sampling_params=full_sampling_params,
            mol2_path=mol2_path,
            positioning_strategy=positioning_strategy,
            device=device,
            logger=logger,
            return_trajectory=True
        )
    else:
        # Regular molecule generation - run once and get full trajectory
        pos_list, atom_list, pos_traj_list, atom_traj_list = trajectory_with_model(
            model=model,
            dataset_info=dataset_info,
            num_samples=num_samples,
            batch_size=batch_size,
            sampling_params=full_sampling_params,
            device=device,
            logger=logger,
            return_trajectory=True
        )
        update_masks_list = [None] * len(pos_list)  # No update masks for regular generation
    
    # Extract positions at specific timesteps from the full trajectory
    for mol_idx in range(len(pos_list)):
        trajectory_data[mol_idx] = {}
        
        pos_traj = pos_traj_list[mol_idx]  # Full trajectory for this molecule
        atom_traj = atom_traj_list[mol_idx]  # Full trajectory for this molecule
        update_mask = update_masks_list[mol_idx] if mol_idx < len(update_masks_list) else None
        
        for timestep in timesteps:
            # Convert timestep to trajectory index
            # pos_traj[0] = t=max_timestep, pos_traj[-1] = t=0
            if timestep == 0:
                traj_idx = len(pos_traj) - 1  # Last index = t=0
            else:
                # Linear interpolation: timestep -> trajectory index
                traj_idx = int((max_timestep - timestep) * len(pos_traj) / max_timestep)
                traj_idx = max(0, min(traj_idx, len(pos_traj) - 1))  # Clamp to valid range
            
            pos = pos_traj[traj_idx]
            atom_type = atom_traj[traj_idx]
            
            # Process atom types (remove charge and get indices)
            import torch
            if hasattr(atom_type, 'cpu'):
                atom_type = atom_type.cpu()
            if len(atom_type.shape) > 1 and atom_type.shape[1] > 1:
                atom_type = atom_type[:, :-1]  # Remove charge
                atom_type = torch.argmax(atom_type, dim=1)
            
            trajectory_data[mol_idx][timestep] = (pos, atom_type, update_mask)
    
    if logger:
        logger.info(f"Extracted trajectory data for {len(trajectory_data)} molecules at {len(timesteps)} timesteps")
    
    return trajectory_data 