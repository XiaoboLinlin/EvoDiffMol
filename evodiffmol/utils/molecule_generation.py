#!/usr/bin/env python3

import torch
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from rdkit import Chem

# PyTorch Geometric imports
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, unbatch

# Local imports
from utils.sample import adjacent_matrix, construct_dataset, construct_dataset_with_fixed
from utils.reconstruct import build_molecule, mol2smiles
from utils.mol2_parser import parse_mol2_fixed_structure, create_update_mask

logger = logging.getLogger(__name__)


def extract_largest_connected_component_3d(positions, atom_types, dataset_info, distance_threshold=1.8):
    """
    Extract the largest connected component from 3D coordinates before building the molecule.
    This helps avoid Open Babel failures due to disconnected fragments.
    
    Args:
        positions: 3D coordinates (N x 3 tensor)
        atom_types: Atom type indices (N tensor)
        dataset_info: Dataset information
        distance_threshold: Distance threshold for connectivity (default: 1.8 Å)
    
    Returns:
        Tuple of (largest_positions, largest_atom_types, component_indices)
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components
    
    # Convert to numpy
    pos_np = positions.detach().cpu().numpy() if hasattr(positions, 'detach') else np.array(positions)
    
    # Calculate pairwise distances
    distances = pdist(pos_np)
    dist_matrix = squareform(distances)
    
    # Create adjacency matrix based on distance threshold
    adjacency = dist_matrix < distance_threshold
    
    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)
    
    if n_components == 1:
        # Single component, return everything
        return positions, atom_types, list(range(len(positions)))
    
    # Find the largest component
    component_sizes = np.bincount(labels)
    largest_component_label = np.argmax(component_sizes)
    
    # Extract indices of the largest component
    largest_component_indices = np.where(labels == largest_component_label)[0]
    
    # Extract corresponding positions and atom types
    largest_positions = positions[largest_component_indices]
    largest_atom_types = atom_types[largest_component_indices]
    
    logger.debug(f"Extracted largest component: {len(largest_component_indices)}/{len(positions)} atoms")
    
    return largest_positions, largest_atom_types, largest_component_indices.tolist()


def generate_molecules_with_model(model, dataset_info, num_samples: int, batch_size: int, 
                                sampling_params: Dict[str, Any], context=None, device='cuda', model_config=None, logger=None, build_method='openbabel'):
    """
    Generate molecules using a trained model with consistent logic.
    
    Args:
        model: Trained model for molecule generation
        dataset_info: Dataset configuration
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        sampling_params: Dictionary of sampling parameters
        context: Optional context for conditional generation
        device: Device to run generation on
        model_config: Optional model configuration (for num_diffusion_timesteps)
        logger: Optional logger for output (if None, uses module logger)
        build_method: Method for building molecules ('openbabel' or 'basic')
    
    Returns:
        Tuple of (position_list, atom_type_list, smile_list, valid, stable)
    """
    # Use passed logger or fall back to module logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Construct dataset for sampling
    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)
    
    
    position_list = []
    atom_type_list = []
    total_smile_list = []
    smile_list = []
    valid = 0
    stable = 0
    attempted_molecules = 0  # Track total molecules attempted (including failed batches)
    total_generated_molecules = 0  # Track molecules that came out of diffusion (before validation)
    
    model.eval()
    
    with torch.no_grad():
        for n, datas in enumerate(tqdm(data_list, desc='Generating molecules')):
            batch = Batch.from_data_list(datas).to(device)
            batch_size_actual = len(datas)  # Actual molecules in this batch
            attempted_molecules += batch_size_actual  # Count all attempted molecules
            
            clip_local = None
            retry_count = 0
            max_retries = 3
            
            FINISHED = True
            while FINISHED and retry_count < max_retries:
                try:
                    # Generate molecules using model sampling
                    pos_gen, _, atom_type, _ = model.langevin_dynamics_sample(
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
                        clip_local=clip_local,
                        sampling_type=sampling_params.get('sampling_type', 'generalized'),
                        eta=sampling_params.get('eta', 1.0),
                    )
                    
                    # Unbatch the generated molecules
                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_type_list_batch = unbatch(atom_type, batch.batch)
                    current_batch_size = len(pos_list)
                    
                    # Process each molecule in the batch
                    for m in range(current_batch_size):
                        pos = pos_list[m]
                        atom_type = atom_type_list_batch[m]

                        # charge
                        atom_type = atom_type[:, :-1]
                        atom_type = torch.argmax(atom_type, dim=1)
                        
                        # Count every molecule that came out of diffusion
                        total_generated_molecules += 1
                        
                        mol = build_molecule(pos, atom_type, dataset_info, method=build_method)

                        if mol is None:
                            print("here")
                        smile = mol2smiles(mol)
                        total_smile_list.append(smile)

                        if smile is not None:
                            valid += 1
                            if "." not in smile:
                                stable += 1
                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                            mol = largest_mol
                            canonical_smile = mol2smiles(mol)
                            smile_list.append(canonical_smile)

                            position_list.append(pos.cpu().detach())
                            atom_type_list.append(atom_type.cpu().detach())
                    FINISHED = False  # Successfully completed
                except FloatingPointError:
                    retry_count += 1
                    if retry_count == 1:
                        clip_local = 10
                        logger.warning(f'NaN detected during sampling. Retrying with local clipping (clip_local=10). Attempt {retry_count}/{max_retries}')
                    elif retry_count == 2:
                        clip_local = 5
                        logger.warning(f'NaN still occurring. Retrying with stronger clipping (clip_local=5). Attempt {retry_count}/{max_retries}')
                    elif retry_count == 3:
                        clip_local = 2
                        logger.warning(f'NaN persisting. Final retry with very strong clipping (clip_local=2). Attempt {retry_count}/{max_retries}')
                    else:
                        logger.error(f'Failed to generate batch after {max_retries} attempts. Skipping this batch of {batch_size_actual} molecules.')
                        # Don't subtract from attempted_molecules - they were attempted but failed
                        FINISHED = False  # Give up on this batch
    # Log batch statistics
    successfully_generated = len(position_list)  # This equals 'valid' by design
    
    if successfully_generated > 0:
        assert len(position_list) == len(atom_type_list) == len(smile_list), \
            f"List length mismatch: positions={len(position_list)}, atoms={len(atom_type_list)}, smiles={len(smile_list)}"
        assert successfully_generated == valid, \
            f"Logic error: successfully_generated ({successfully_generated}) should equal valid ({valid})"
        
        logger.info("----------------------------")
        logger.info(f"Generation Summary:")
        logger.info(f"  - Requested: {num_samples} molecules")
        logger.info(f"  - Attempted: {attempted_molecules} molecules")
        logger.info(f"  - Generated by diffusion: {total_generated_molecules} molecules")
        logger.info(f"  - Valid molecules: {valid}/{total_generated_molecules} ({valid/total_generated_molecules:.4f})")
        logger.info(f"  - Failed molecules: {total_generated_molecules - valid}")
        logger.info(f"  - Stable (no fragments): {stable}/{valid} ({stable/valid:.4f})")
        logger.info(f"  - Unique: {len(set(smile_list))}/{len(smile_list)} ({len(set(smile_list))/len(smile_list) if len(smile_list) > 0 else 0:.4f})")
    else:
        logger.warning("No molecules generated in this session - all batches failed")
        logger.info("----------------------------")
        logger.info(f"Generation Summary:")
        logger.info(f"  - Requested: {num_samples} molecules")
        logger.info(f"  - Attempted: {attempted_molecules} molecules") 
        logger.info(f"  - Successfully generated: 0 molecules")
        
    assert len(position_list) == len(atom_type_list) == len(smile_list), \
        f"Final list length mismatch: positions={len(position_list)}, atoms={len(atom_type_list)}, smiles={len(smile_list)}"
    return position_list, atom_type_list, smile_list, valid, stable


def generate_molecules_to_population(
    model: Any,
    dataset_info: dict,
    num_samples: int,
    batch_size: int,
    sampling_params: dict,
    context: Optional[List[str]] = None,
    device: str = 'cuda',
    model_config: Any = None,
    log_prefix: str = "Molecule generation"
) -> List[Data]:
    """
    Generate molecules using the shared generation utility and convert to Data objects.
    
    This is a shared function used by both the trainer and population manager to ensure
    consistent molecule generation behavior.
    
    Args:
        model: Diffusion model for molecule generation
        dataset_info: Dataset information for molecule reconstruction
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        sampling_params: Sampling parameters for the model
        context: Context for conditional generation
        device: Computing device
        model_config: Model configuration
        log_prefix: Prefix for logging messages
        
    Returns:
        List of generated Data objects
    """
    
    # Generate molecules using shared logic
    population_logger = logging.getLogger(__name__)
    position_list, atom_type_list, smile_list, valid, stable = generate_molecules_with_model(
        model=model,
        dataset_info=dataset_info,
        num_samples=num_samples,
        batch_size=batch_size,
        sampling_params=sampling_params,
        context=context,
        device=device,
        model_config=model_config,
        logger=population_logger
    )
    
    # Validate that all lists have the same length
    assert len(position_list) == len(atom_type_list) == len(smile_list), \
        f"List length mismatch: positions={len(position_list)}, atoms={len(atom_type_list)}, smiles={len(smile_list)}"
    
    # Process results using the main lists
    population = []
    for i, (pos, atom_type, smile) in enumerate(zip(position_list, atom_type_list, smile_list)):
        if smile is not None:  # Only create objects for valid molecules
            # Convert atom_type indices back to atomic numbers
            # Create reverse mapping from dataset_info['atom_index']
            atom_index_reverse = {v: k for k, v in dataset_info['atom_index'].items()}
            atom_type_atomic_numbers = torch.tensor([atom_index_reverse[int(idx)] for idx in atom_type])
            
            # adj = adjacent_matrix(len(atom_type))
            data = Data(
                pos=pos,
                atom_type=atom_type_atomic_numbers,  # Convert to actual atomic numbers
                # num_atoms=torch.tensor([len(atom_type)], dtype=torch.long),
                # num_nodes_per_graph=torch.tensor([len(atom_type)], dtype=torch.long),
                atom_type_index = atom_type,
                # edge_index=torch.empty((2, 0), dtype=torch.long),  # Empty edge_index for compatibility
                # edge_type=torch.empty((0,), dtype=torch.long), # Empty edge_type for compatibility
                smiles=smile
            )
            
            population.append(data)
    
    # Log generation statistics using shared utility
    logger.info(f"Successfully generated {len(population)} molecules")
    from utils.evaluation_utils import log_generation_statistics
    log_generation_statistics(position_list, atom_type_list, smile_list, valid, stable, logger, log_prefix)
    
    return population[:num_samples]  # Return exactly the requested size


def generate_molecules_with_fixed_structure(model, dataset_info, num_samples, batch_size, sampling_params, 
                                           mol2_path, context=None, device='cuda', model_config=None, remove_h=True,
                                           positioning_strategy='plane_through_origin', build_method='openbabel'):
    """
    Generate molecules with a fixed substructure from MOL2 file
    
    Args:
        model: Trained diffusion model
        dataset_info (dict): Dataset configuration and statistics
        num_samples (int): Total number of molecules to generate
        batch_size (int): Batch size for generation
        sampling_params (dict): Sampling hyperparameters
        mol2_path (str): Path to MOL2 file containing fixed structure
        context: Additional context (unused for now)
        device (str): Device to run on
        model_config: Model configuration
        remove_h (bool): Whether to remove hydrogens
        positioning_strategy (str): How to position the scaffold:
            - 'plane_through_origin': Rotate and shift scaffold plane but keep plane through (0,0,0)
            - 'free_position': Completely free positioning anywhere in space  
            - 'sphere_constraint': Position within a sphere around origin
            - 'center_only': Only center at origin with random rotation (no translation)
        build_method (str): Method for building molecules ('openbabel' or 'basic')
        
    Returns:
        tuple: (position_list, atom_type_list, smile_list, valid, stable, update_masks_list)
    """
    # Parse fixed structure from MOL2
    logger.info(f"Parsing fixed structure from {mol2_path}")
    fixed_structure = parse_mol2_fixed_structure(mol2_path, remove_h=remove_h, dataset_info=dataset_info)
    
    logger.info(f"Fixed structure: {fixed_structure['num_fixed_atoms']} atoms")
    logger.info(f"Fixed atom types: {fixed_structure['fixed_atom_types'].tolist()}")
    logger.info(f"Using positioning strategy: {positioning_strategy}")
    
    # Construct dataset with fixed structure (molecule sizes sampled from dataset distribution)
    logger.info("Constructing dataset with fixed structure and sampled molecule sizes")
    data_list, total_samples, update_masks = construct_dataset_with_fixed(
        num_samples=num_samples, 
        batch_size=batch_size, 
        dataset_info=dataset_info, 
        fixed_structure=fixed_structure,
        positioning_strategy=positioning_strategy
    )
    
    logger.info(f"Generated {total_samples} data samples for fixed structure generation")
    
    model.eval()
    
    position_list = []
    atom_type_list = []
    smile_list = []
    update_masks_list = []
    valid = 0
    stable = 0
    total_generated_molecules = 0  # Track molecules that came out of diffusion (before validation)
    
    # Generate molecules with fixed structure
    logger.info("Starting fixed structure generation")
    with tqdm(data_list, desc="Generating molecules with fixed structure") as pbar:
        for batch_idx, (batch_data_list, batch_update_masks_list) in enumerate(zip(data_list, update_masks)):
            try:
                # Convert list of Data objects to Batch
                from torch_geometric.data import Batch
                batch_data = Batch.from_data_list(batch_data_list).to(device)
                
                # Convert list of update masks to single tensor for batch
                if batch_update_masks_list is not None:
                    batch_update_masks = torch.cat(batch_update_masks_list, dim=0).to(device)
                else:
                    batch_update_masks = None
                
                current_batch_size = batch_data.num_graphs
                
                # Run diffusion sampling
                pos_gen, _, atom_type, _ = model.langevin_dynamics_sample(
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
                    update_mask=batch_update_masks  # Pass update mask for fixed structure preservation
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

                    # charge
                    atom_type = atom_type[:, :-1]
                    atom_type = torch.argmax(atom_type, dim=1)
                    
                    # Count every molecule that came out of diffusion
                    total_generated_molecules += 1
                    
                    # # Verify that fixed atoms haven't moved significantly (optional verification)
                    # fixed_indices = (mol_update_mask == 0).nonzero(as_tuple=True)[0]
                    # if len(fixed_indices) > 0:
                    #     # We could add drift checking here if needed, but it's no longer critical
                    #     pass
                    
                    # Build molecule and generate SMILES
                    mol = build_molecule(pos, atom_type, dataset_info, method=build_method)
                    if mol is None:
                        continue
                        
                    smile = mol2smiles(mol)
                    if smile is not None:
                        valid += 1
                        if "." not in smile:
                            stable += 1
                        position_list.append(pos.detach().cpu())
                        atom_type_list.append(atom_type.detach().cpu())
                        smile_list.append(smile)
                        update_masks_list.append(mol_update_mask.detach().cpu() if mol_update_mask is not None else None)
                        
                        
                FINISHED = False
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
            
            pbar.update(1)
            pbar.set_postfix({'valid': valid, 'stable': stable})
    
    logger.info(f"Generated {valid} valid molecules with fixed structure")
    logger.info(f"Total generated by diffusion: {total_generated_molecules}")
    logger.info(f"Valid molecules: {valid}/{total_generated_molecules} ({valid/total_generated_molecules:.2%})")
    logger.info(f"Stable molecules: {stable}/{total_generated_molecules} ({stable/total_generated_molecules:.2%})")
    
    return position_list, atom_type_list, smile_list, valid, stable, update_masks_list


def generate_molecules_with_fixed_structure_to_population(
    model, dataset_info, num_samples: int, batch_size: int, 
    sampling_params: Dict[str, Any], mol2_path: str, context=None, device='cuda', 
    model_config=None, remove_h=True, positioning_strategy='plane_through_origin',
    log_prefix: str = "Fixed structure generation"
) -> List[Data]:
    """
    Generate molecules with fixed structure and return them as population Data objects.
    
    Args:
        model: Trained diffusion model
        dataset_info: Dataset configuration
        num_samples: Number of molecules to generate
        batch_size: Batch size for generation
        sampling_params: Sampling parameters
        mol2_path: Path to MOL2 file with fixed structure
        context: Optional context
        device: Device to run on
        model_config: Model configuration
        remove_h: Whether to remove hydrogens
        positioning_strategy: Scaffold positioning strategy
        log_prefix: Logging prefix
        
    Returns:
        List of Data objects for population use
    """
    # Generate molecules with fixed structure
    position_list, atom_type_list, smile_list, valid, stable, update_masks_list = generate_molecules_with_fixed_structure(
        model=model,
        dataset_info=dataset_info,
        num_samples=num_samples,
        batch_size=batch_size,
        sampling_params=sampling_params,
        mol2_path=mol2_path,
        context=context,
        device=device,
        model_config=model_config,
        remove_h=remove_h,
        positioning_strategy=positioning_strategy
    )
    
    # Convert to Data objects for population use
    population_molecules = []
    
    logger.info(f"{log_prefix}: Converting {len(position_list)} molecules to population format")
    
    if len(position_list) == 0:
        logger.error(f"{log_prefix}: No valid molecules generated from scaffold {mol2_path}")
        logger.error(f"Valid molecules: {valid}/{num_samples}")
        logger.error(f"Stable molecules: {stable}/{num_samples}")
        return []
    
    for i, (pos, atom_type, smile, update_mask) in enumerate(zip(
        position_list, atom_type_list, smile_list, update_masks_list
    )):
        try:
            # Convert atom_type indices back to atomic numbers (like regular generation)
            atom_index_reverse = {v: k for k, v in dataset_info['atom_index'].items()}
            atom_type_atomic_numbers = torch.tensor([atom_index_reverse[int(idx)] for idx in atom_type])
            
            # Create Data object (match regular generation format closely)
            data = Data(
                pos=pos.float(),
                atom_type=atom_type_atomic_numbers,  # Add atomic numbers for transforms
                atom_type_index=atom_type,          # Keep indices for compatibility  
                smiles=smile
            )
            
            # Add update mask information if available
            if update_mask is not None:
                data.update_mask = update_mask.float()
            
            population_molecules.append(data)
            
        except Exception as e:
            logger.warning(f"Failed to convert molecule {i} to population format: {e}")
            continue
    
    logger.info(f"{log_prefix}: Successfully created {len(population_molecules)} population molecules")
    return population_molecules




 