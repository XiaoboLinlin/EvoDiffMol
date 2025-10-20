from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tqdm import tqdm
from .common import *
import logging

logger = logging.getLogger(__name__)


def adjacent_matrix(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    # print(n_particles)
    rows = torch.LongTensor(rows).unsqueeze(0)
    cols = torch.LongTensor(cols).unsqueeze(0)
    # print(rows.size())
    adj = torch.cat([rows, cols], dim=0)
    return adj


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        # print(self.normalizer)
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


n_nodes = {26: 4711, 31: 3365, 19: 3093, 22: 3344, 32: 3333, 25: 4533,
           36: 1388, 23: 4375, 33: 2686, 29: 3242, 14: 2469, 28: 4838,
           41: 630, 9: 1858, 18: 2621, 27: 5417, 10: 2865, 30: 3605,
           42: 502, 13: 2164, 11: 3051, 21: 4493, 15: 2292, 12: 2900,
           40: 691, 45: 184, 20: 4883, 24: 3716, 46: 213, 39: 752,
           17: 2446, 16: 3094, 35: 1879, 38: 915, 44: 691, 43: 360,
           50: 37, 8: 1041, 7: 655, 34: 2168, 47: 119, 49: 73, 6: 705,
           37: 928, 51: 21, 4: 45, 48: 187, 5: 111, 52: 42, 54: 93,
           56: 12, 57: 8, 55: 35, 71: 1, 61: 9, 58: 18, 59: 5, 67: 28,
           3: 4, 65: 2, 63: 5, 62: 1, 86: 1, 66: 20, 106: 2, 53: 3, 77: 1, 68: 1, 98: 1}

qm9_noh_n_nodes = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                   15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                   8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1}

MAX_NODES = 29


def construct_dataset(num_sample, batch_size, dataset_info):
    nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    data_list = []

    num_atom = len(dataset_info['atom_decoder']) + 1  # charge
    # num_atom = 20
    nodesxsample_list = []
    
    # Calculate number of full batches and remainder
    num_full_batches = int(num_sample / batch_size)
    remainder = num_sample % batch_size
    
    # Process full batches
    for _ in tqdm(range(num_full_batches)):
        datas = []
        nodesxsample = nodes_dist.sample(batch_size).tolist()
        nodesxsample_list.append(nodesxsample)
        # atom_type_list = torch.randn(batch_size,MAX_NODES,6)
        # pos_list = torch.randn(batch_size,MAX_NODES,3)
        for n_particles in nodesxsample:
            # n_particles = 19
            atom_type = torch.randn(n_particles, num_atom)
            pos = torch.randn(n_particles, 3)
            # atom_type = torch.zeros(n_particles, num_atom).uniform_(-3,+3)
            # pos = torch.zeros(n_particles, 3).uniform_(-3,+3)
            # atom_type = torch.randn(MAX_NODES, 5)[:n_particles,:]
            # pos = torch.randn(MAX_NODES, 3)[:n_particles,:]
            # atom_type = atom_type_list[i,:n_particles,:].squeeze(0)
            # pos = pos_list[i,:n_particles,:].squeeze(0)

            # coors = pos
            adj = adjacent_matrix(n_particles)
            data = Data(x=atom_type, edge_index=adj, pos=pos)
            datas.append(data)
        data_list.append(datas)
    
    # Process remainder batch if it exists
    if remainder > 0:
        datas = []
        nodesxsample = nodes_dist.sample(remainder).tolist()
        nodesxsample_list.append(nodesxsample)
        for n_particles in nodesxsample:
            atom_type = torch.randn(n_particles, num_atom)
            pos = torch.randn(n_particles, 3)
            adj = adjacent_matrix(n_particles)
            data = Data(x=atom_type, edge_index=adj, pos=pos)
            datas.append(data)
        data_list.append(datas)
        
    return data_list, nodesxsample_list


def apply_random_rotation(positions):
    """
    Apply random 3D rotation to positions using Rodrigues' rotation formula
    
    Args:
        positions (torch.Tensor): Input positions (N, 3)
        
    Returns:
        torch.Tensor: Rotated positions (N, 3)
    """
    # Generate random rotation axis and angle
    rotation_axis = torch.randn(3)
    rotation_axis = rotation_axis / torch.norm(rotation_axis)
    rotation_angle = torch.rand(1) * 2 * 3.14159  # Random angle [0, 2π]
    
    # Rodrigues' rotation formula
    cos_angle = torch.cos(rotation_angle)
    sin_angle = torch.sin(rotation_angle)
    axis = rotation_axis
    
    # Rotation matrix
    rotation_matrix = cos_angle * torch.eye(3) + sin_angle * torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]], 
        [-axis[1], axis[0], 0]
    ]) + (1 - cos_angle) * torch.outer(axis, axis)
    
    # Apply rotation
    rotated_positions = torch.matmul(positions, rotation_matrix.T)
    return rotated_positions


def randomize_scaffold_position(fixed_positions, mol_target_atoms, strategy='plane_through_origin'):
    """
    Randomize the position and orientation of a fixed scaffold structure.
    
    Args:
        fixed_positions (torch.Tensor): Initial scaffold positions (N, 3)
        mol_target_atoms (int): Target total number of atoms in the molecule
        strategy (str): Positioning strategy
            - 'plane_through_origin': Rotate and shift scaffold plane but keep plane through (0,0,0)
            - 'free_position': Completely free positioning anywhere in space
            - 'sphere_constraint': Position within a sphere around origin
            - 'center_only': Only center at origin (no rotation or translation)
    
    Returns:
        torch.Tensor: Randomized scaffold positions (N, 3)
    """
    # 1. Center the scaffold at origin first (remove original center)
    fixed_center = fixed_positions.mean(dim=0)
    centered_positions = fixed_positions - fixed_center

    # 2. Apply random rotation around origin (skip for center_only strategy)
    if strategy == 'center_only':
        rotated_positions = centered_positions  # No rotation for center_only
    else:
        rotated_positions = apply_random_rotation(centered_positions)

    # 3. Apply positioning strategy
    if strategy == 'plane_through_origin':
        # Original logic: shift within plane while keeping plane through origin
        if len(rotated_positions) >= 3:
            # Get two vectors in the plane
            v1 = rotated_positions[1] - rotated_positions[0]
            v2 = rotated_positions[2] - rotated_positions[0]
            # Calculate normal vector (perpendicular to plane)
            normal = torch.linalg.cross(v1, v2)
            normal = normal / (torch.norm(normal) + 1e-8)
            
            # Generate two orthogonal vectors in the scaffold plane
            plane_vec1 = v1 / (torch.norm(v1) + 1e-8)
            plane_vec2 = torch.linalg.cross(normal, plane_vec1)
            plane_vec2 = plane_vec2 / (torch.norm(plane_vec2) + 1e-8)
            
            # Random shift within the plane
            base_shift = 0.8 + 0.15 * mol_target_atoms
            magnitude_variation = 0.5 + 0.5 * torch.rand(1)
            shift_magnitude = base_shift * magnitude_variation
            shift_direction = torch.rand(1) * 2 * 3.14159
            
            shift_vector = shift_magnitude * (torch.cos(shift_direction) * plane_vec1 + 
                                            torch.sin(shift_direction) * plane_vec2)
            final_positions = rotated_positions + shift_vector
        else:
            # Fallback for molecules with < 3 atoms
            shift_magnitude = 1 + 0.05 * mol_target_atoms
            random_shift = torch.randn(3) * shift_magnitude
            final_positions = rotated_positions + random_shift
            
    elif strategy == 'free_position':
        # Complete freedom: position anywhere in 3D space
        # Scale translation based on molecule size for realistic placement
        base_translation = 1.5 + 0.2 * mol_target_atoms  # Larger range than plane constraint
        random_translation = torch.randn(3) * base_translation
        final_positions = rotated_positions + random_translation
        
    elif strategy == 'sphere_constraint':
        # Position within a sphere around origin
        base_radius = 1.8 + 0.3 * mol_target_atoms  # More moderate scaling
        # Add variation to the radius like other strategies
        magnitude_variation = 0.5 + 0.5 * torch.rand(1)
        sphere_radius = base_radius * magnitude_variation
        # Generate random point on sphere surface first
        random_direction = torch.randn(3)
        random_direction = random_direction / torch.norm(random_direction)
        # Random distance from center (0 to sphere_radius)
        random_distance = torch.rand(1) * sphere_radius
        sphere_translation = random_direction * random_distance
        final_positions = rotated_positions + sphere_translation
        
    elif strategy == 'center_only':
        # No rotation or translation (scaffold center remains at origin in original orientation)
        final_positions = rotated_positions
        
    else:
        raise ValueError(f"Unknown positioning strategy: {strategy}")
    
    return final_positions


def create_molecule_data(fixed_structure, dataset_info, mol_target_atoms, positioning_strategy='plane_through_origin'):
    """
    Create a single molecule's data with fixed structure and random new atoms
    
    Args:
        fixed_structure (dict): Fixed structure information from MOL2
        dataset_info (dict): Dataset configuration  
        mol_target_atoms (int): Target total number of atoms for this molecule
        positioning_strategy (str): How to position the scaffold ('plane_through_origin', 'free_position', 'sphere_constraint', 'center_only')
        
    Returns:
        tuple: (data, update_mask, fixed_positions)
    """
    from .mol2_parser import create_update_mask
    
    num_fixed_atoms = fixed_structure['num_fixed_atoms']
    num_new_atoms = mol_target_atoms - num_fixed_atoms
    num_atom_features = len(dataset_info['atom_decoder']) + 1  # charge
    
    # Create combined atom features: fixed + random new atoms
    fixed_atom_features = torch.zeros(num_fixed_atoms, num_atom_features)
    # Set fixed atom types in one-hot format
    for i, atom_type_idx in enumerate(fixed_structure['fixed_atom_types']):
        # atom_type_idx is already the dataset index (0, 1, 2, etc.)
        # We just need to set the corresponding one-hot position
        if atom_type_idx < len(dataset_info['atom_decoder']):
            fixed_atom_features[i, atom_type_idx] = 1.0
        else:
            raise ValueError(f"Invalid atom type index {atom_type_idx} for dataset with {len(dataset_info['atom_decoder'])} atom types")
    
    # Random features for new atoms
    new_atom_features = torch.randn(num_new_atoms, num_atom_features)
    
    # Combine fixed and new atom features
    atom_type = torch.cat([fixed_atom_features, new_atom_features], dim=0)
    
    # Randomize fixed structure position and orientation using flexible strategy
    fixed_positions = randomize_scaffold_position(
        fixed_structure['fixed_positions'].clone(), 
        mol_target_atoms, 
        strategy=positioning_strategy
    )
    
    # Random positions for new atoms (distributed around the scaffold)
    scaffold_center = fixed_positions.mean(dim=0)
    scaffold_radius = torch.norm(fixed_positions - scaffold_center, dim=1).max().item()
    
    # Generate new atoms in a reasonable vicinity of the scaffold
    new_positions = []
    for _ in range(num_new_atoms):
        # Random position within 2x scaffold radius from scaffold center
        # random_offset = torch.randn(3) * (scaffold_radius * 2)
        random_offset = torch.randn(3) * (scaffold_radius)
        # random_offset = torch.randn(3)
        new_pos = scaffold_center + random_offset
        new_positions.append(new_pos)
    
    new_positions = torch.stack(new_positions) if new_positions else torch.empty((0, 3))
    
    # Combine all positions
    all_positions = torch.cat([fixed_positions, new_positions], dim=0)
    
    # Create update mask: 0 for fixed atoms, 1 for new atoms
    update_mask = create_update_mask(num_fixed_atoms, mol_target_atoms)
    
    # Create adjacency matrix for all atoms
    adj = adjacent_matrix(mol_target_atoms)
    
    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data
    data = Data(
        x=atom_type,
        pos=all_positions,
        edge_index=adj,
        num_nodes=mol_target_atoms
    )
    
    return data, update_mask, fixed_positions


def sample_molecule_sizes(nodes_dist_info, batch_size, num_fixed_atoms, max_attempts=5):
    """
    Sample molecule sizes from dataset distribution, ensuring they're larger than fixed atoms
    
    Args:
        nodes_dist_info: Dataset n_nodes information (either dict or distribution object)
        batch_size (int): Number of molecules to sample
        num_fixed_atoms (int): Number of fixed atoms (minimum size)
        max_attempts (int): Maximum attempts to resample if size is too small
        
    Returns:
        list: List of molecule sizes
    """
    # Create distribution if nodes_dist_info is a dict
    if isinstance(nodes_dist_info, dict):
        nodes_dist = DistributionNodes(nodes_dist_info)
    else:
        nodes_dist = nodes_dist_info
    
    sampled_sizes = []
    for _ in range(batch_size):
        attempts = 0
        while attempts < max_attempts:
            size = nodes_dist.sample(1)[0]  # Sample single molecule size
            if size > num_fixed_atoms:
                sampled_sizes.append(size)
                break
            attempts += 1
        
        if attempts >= max_attempts:
            # Fallback: use minimum valid size
            sampled_sizes.append(num_fixed_atoms + 1)
            logger.warning(f"Could not sample valid molecule size after {max_attempts} attempts, using fallback size {num_fixed_atoms + 1}")
    
    return sampled_sizes


def process_batch(num_molecules, dataset_info, fixed_structure, positioning_strategy='plane_through_origin'):
    """
    Helper function to process a batch of molecules with fixed structure
    
    Args:
        num_molecules (int): Number of molecules to generate in this batch
        dataset_info (dict): Dataset configuration
        fixed_structure (dict): Fixed structure information
        positioning_strategy (str): Positioning strategy for scaffold placement
        
    Returns:
        tuple: (batch_data_list, batch_update_masks)
    """
    # Sample molecule sizes from dataset distribution
    mol_sizes = sample_molecule_sizes(dataset_info['n_nodes'], num_molecules, fixed_structure['num_fixed_atoms'])
    
    batch_data = []
    batch_update_masks = []
    
    for mol_target_atoms in mol_sizes:
        data, update_mask, _ = create_molecule_data(fixed_structure, dataset_info, mol_target_atoms, positioning_strategy)
        batch_data.append(data)
        batch_update_masks.append(update_mask)
    
    return batch_data, batch_update_masks


def construct_dataset_with_fixed(num_samples, batch_size, dataset_info, fixed_structure, positioning_strategy='plane_through_origin'):
    """
    Construct dataset with fixed structure for sampling
    
    Args:
        num_samples (int): Total number of samples to generate
        batch_size (int): Batch size for processing
        dataset_info (dict): Dataset configuration and statistics
        fixed_structure (dict): Fixed structure information from MOL2 file
        positioning_strategy (str): How to position the scaffold:
            - 'plane_through_origin': Rotate and shift scaffold plane but keep plane through (0,0,0)
            - 'free_position': Completely free positioning anywhere in space  
            - 'sphere_constraint': Position within a sphere around origin
            - 'center_only': Only center at origin with random rotation (no translation)
        
    Returns:
        tuple: (data_list, total_samples, update_masks)
            - data_list: List of batched data for sampling
            - total_samples: Total number of samples that will be generated
            - update_masks: Corresponding update masks for each batch
    """
    logger.info(f"Constructing dataset with fixed structure using '{positioning_strategy}' positioning strategy")
    
    data_list = []
    update_masks = []
    
    # Process full batches
    full_batches = num_samples // batch_size
    for _ in range(full_batches):
        batch_data, batch_update_masks = process_batch(batch_size, dataset_info, fixed_structure, positioning_strategy)
        data_list.append(batch_data)
        update_masks.append(batch_update_masks)
    
    # Process remainder batch if needed
    remainder = num_samples % batch_size
    if remainder > 0:
        batch_data, batch_update_masks = process_batch(remainder, dataset_info, fixed_structure, positioning_strategy)
        data_list.append(batch_data)
        update_masks.append(batch_update_masks)
    
    total_samples = full_batches * batch_size + remainder
    logger.info(f"Generated {len(data_list)} batches with {total_samples} total samples")
    
    return data_list, total_samples, update_masks


# def construct_dataset_pocket(num_sample, batch_size, dataset_info, num_points=None, *protein_information):
#     nodes_dist = DistributionNodes(dataset_info['n_nodes'])
#     data_list = []

#     num_atom = len(dataset_info['atom_decoder']) + 1  # charge
#     # num_atom = 20
#     nodesxsample_list = []
#     protein_atom_feature_full, protein_pos, protein_bond_index = protein_information
#     for n in tqdm(range(int(num_sample / batch_size))):
#         datas = []
#         if num_points is not None:
#             nodesxsample = nodes_dist.sample(batch_size - 1).tolist().append(num_points)
#         else:
#             nodesxsample = nodes_dist.sample(batch_size).tolist()
#         nodesxsample_list.append(nodesxsample)
#         # atom_type_list = torch.randn(batch_size,MAX_NODES,6)
#         # pos_list = torch.randn(batch_size,MAX_NODES,3)
#         for i, n_particles in enumerate(nodesxsample):
#             # n_particles = 19
#             # atom_type = torch.randn(n_particles, num_atom)
#             # pos = torch.randn(n_particles, 3)
#             atom_type = torch.zeros(n_particles, num_atom).uniform_(-3, +3)
#             pos = torch.zeros(n_particles, 3).uniform_(-3, +3)
#             coors = pos
#             adj = adjacent_matrix(n_particles)
#             data = Data(ligand_atom_type=atom_type, ligand_bond_index=adj, ligand_pos=pos,
#                         protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
#                         protein_bond_index=protein_bond_index)
#             datas.append(data)
#         data_list.append(datas)
#     return data_list, nodesxsample_list
