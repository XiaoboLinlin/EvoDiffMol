"""
GA Population Dataset for PyTorch Geometric integration.
"""

import os
import os.path as osp
import logging
from typing import List, Optional, Dict, Any
import tempfile
import shutil

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader

logger = logging.getLogger(__name__)


class GAPopulationDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for genetic algorithm populations.
    
    This dataset handles population data using PyTorch Geometric Data objects,
    enabling proper DataLoader support and efficient batching.
    """
    
    def __init__(self, 
                 population_data: Optional[List[Data]] = None,
                 filepath: Optional[str] = None,
                 transform=None, 
                 pre_transform=None,
                 pre_filter=None):
        """
        Initialize GA Population Dataset.
        
        Args:
            population_data: List of Data objects representing molecules
            filepath: Path to saved population file (.pt format)
            transform: Transform to apply to data objects
            pre_transform: Transform to apply before saving
            pre_filter: Filter to apply before saving
        """
        self.population_data = population_data
        self.filepath = filepath
        
        # Create a temporary directory to be used as the root for the
        # InMemoryDataset. This is a workaround to prevent the dataset
        # from creating `raw` and `processed` folders in the current
        # working directory when `root` is `None`.
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize the parent class with the temporary directory.
            super().__init__(root=temp_dir, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        finally:
            # Ensure the temporary directory is always cleaned up.
            shutil.rmtree(temp_dir)

        # Manually handle data loading or processing.
        if filepath and os.path.exists(filepath):
            # If a filepath is provided, load the pre-processed data directly.
            self.data, self.slices = torch.load(filepath, map_location='cpu', weights_only=False)
        elif population_data is not None:
            self._process()
        else:
            # Initialize empty dataset
            self.data = None
            self.slices = None
    
    @property
    def raw_file_names(self):
        """Raw file names (not used for this dataset)."""
        return []
    
    # @property
    # def processed_file_names(self):
    #     """Processed file names."""
    #     if self.filepath:
    #         return [osp.basename(self.filepath)]
    #     else:
    #         # When no filepath is provided, use a default name
    #         return ['ga_population.pt']
    
    def download(self):
        """Download method (not used for this dataset)."""
        pass
    
    def _process(self):
        """Process population data into PyTorch Geometric format."""
        if self.population_data is None:
            raise ValueError("No population data provided for processing")
        
        if len(self.population_data) == 0:
            raise ValueError("Empty population data provided - molecule generation may have failed")
        
        logger.info(f"Processing {len(self.population_data)} molecules...")
        
        # CRITICAL FIX: Ensure all Data objects have update_mask for collate compatibility
        # This is required when mixing scaffold-based and non-scaffold molecules
        import torch
        for data in self.population_data:
            if not hasattr(data, 'update_mask') or data.update_mask is None:
                # Add default update_mask (all atoms variable)
                if hasattr(data, 'atom_type'):
                    data.update_mask = torch.ones(len(data.atom_type), dtype=torch.float32)
                elif hasattr(data, 'pos'):
                    data.update_mask = torch.ones(len(data.pos), dtype=torch.float32)
        
        # Apply pre_transform if specified
        if self.pre_transform is not None:
            self.population_data = [self.pre_transform(data) for data in self.population_data]
        
        # Collate data for efficient batching
        self.data, self.slices = self.collate(self.population_data)
        
    
    def save_population(self, filepath: str, population: Optional[List[Data]] = None) -> None:
        """
        Save population to a specific filepath.
        
        Args:
            filepath: Path to save the population
            population: Optional population list. If None, saves the dataset's own data.
        """
        # Create directory if it doesn't exist (handle case where filepath is just filename)
        directory = osp.dirname(filepath)
        if directory:  # Only create directory if there is one
            os.makedirs(directory, exist_ok=True)
        
        # if population is not None:
        #     # Save provided population
        #     # Create temporary dataset to get the collated data
        #     temp_dataset = GAPopulationDataset(population_data=population)
        #     torch.save((temp_dataset.data, temp_dataset.slices), filepath)
        #     logger.info(f"Saved population ({len(population)} molecules) to {filepath}")
        # else:
        #     # Save dataset's own data
        torch.save((self.data, self.slices), filepath)
        logger.info(f"Saved population to {filepath}")
    
    @classmethod
    def load_population(cls, filepath: str, transform=None) -> 'GAPopulationDataset':
        """
        Load population from file.
        
        Args:
            filepath: Path to population file
            transform: Transform to apply
            
        Returns:
            GAPopulationDataset instance
        """
        return cls(filepath=filepath, transform=transform)
    
    @staticmethod
    def _extract_numeric_value(attr_val):
        """Extract numeric value from attribute (tensor or float)."""
        if attr_val is None:
            return None
        try:
            return attr_val.item() if hasattr(attr_val, 'item') else float(attr_val)
        except:
            return None
    
    @staticmethod
    def _is_property_attribute(attr_name: str, attr_val) -> bool:
        """Check if attribute is a valid numeric property."""
        # Skip internal/structural attributes
        if attr_name.startswith('_') or attr_name in {
            'atom_type', 'pos', 'edge_index', 'edge_attr', 'batch', 
            'smiles', 'fitness_score', 'num_atoms', 'num_nodes', 'num_edges',
            'x', 'y', 'train_mask', 'val_mask', 'test_mask'
        }:
            return False
        
        # Check if it's numeric
        if callable(attr_val):
            return False
        
        try:
            _ = attr_val.item() if hasattr(attr_val, 'item') else float(attr_val)
            return True
        except:
            return False
    
    @classmethod
    def _detect_property_names(cls, population_sample: List[Data]) -> set:
        """Detect all property attribute names from sample."""
        all_properties = set()
        for data in population_sample:
            # Use dict comprehension for efficiency
            properties = {
                attr_name for attr_name in dir(data)
                if cls._is_property_attribute(attr_name, getattr(data, attr_name, None))
            }
            all_properties.update(properties)
        return all_properties
    
    @classmethod
    def _calculate_property_stats(cls, population: List[Data], prop_name: str) -> Dict[str, float]:
        """Calculate statistics for a single property across population."""
        # Extract all values in one pass (list comprehension)
        values = [
            cls._extract_numeric_value(getattr(data, prop_name, None))
            for data in population if hasattr(data, prop_name)
        ]
        # Filter out None values
        values = [v for v in values if v is not None]
        
        if not values:
            return {}
        
        return {
            f'{prop_name}_mean': sum(values) / len(values),
            f'{prop_name}_min': min(values),
            f'{prop_name}_max': max(values),
        }
    
    def get_population_statistics(self, population: Optional[List[Data]] = None) -> Dict[str, Any]:
        """Get statistics about the population.
        
        Args:
            population: Optional population list. If None, uses the dataset's own data.
        """
        # Convert dataset to list if needed
        if population is None:
            if not hasattr(self, 'data') or self.data is None:
                return {}
            population = [self[i] for i in range(len(self))]
        
        if not population:
            return {}
        
        num_molecules = len(population)
        
        # Basic statistics (vectorized)
        num_atoms_list = [len(data.atom_type) for data in population]
        
        stats = {
            'size': num_molecules,
            'num_atoms_mean': sum(num_atoms_list) / num_molecules,
            'num_atoms_min': min(num_atoms_list) if num_atoms_list else 0,
            'num_atoms_max': max(num_atoms_list) if num_atoms_list else 0,
        }
        
        # Fitness statistics (vectorized extraction)
        fitness_scores = [
            self._extract_numeric_value(data.fitness_score)
            for data in population
            if hasattr(data, 'fitness_score')
        ]
        fitness_scores = [f for f in fitness_scores if f is not None]
        
        if fitness_scores:
            stats.update({
                'fitness_mean': sum(fitness_scores) / len(fitness_scores),
                'fitness_min': min(fitness_scores),
                'fitness_max': max(fitness_scores),
                'valid_fitness_count': len(fitness_scores)
            })
        
        # Automatically detect all property attributes (sample-based)
        sample_size = min(10, num_molecules)
        all_property_names = self._detect_property_names(population[:sample_size])
        
        # Calculate statistics for all detected properties (vectorized)
        for prop_name in sorted(all_property_names):
            prop_stats = self._calculate_property_stats(population, prop_name)
            stats.update(prop_stats)
        
        return stats
    
    @classmethod
    def from_population_list_to_class(cls, population: List[Data], pre_transform) -> 'GAPopulationDataset':
        """
        Create GAPopulationDataset from a list of Data objects.
        
        Args:
            population: List of Data objects
            transform: Transform to apply on-the-fly
            pre_transform: Transform to apply during dataset creation
            
        Returns:
            GAPopulationDataset instance
        """
        return cls(population_data=population, pre_transform=pre_transform)
    
    def to_population_list(self) -> List[Data]:
        """
        Convert dataset back to a list of Data objects, removing bulky attributes
        that were added by transforms and are no longer needed for GA logic.
        
        Returns:
            List of cleaned Data objects.
        """
        cleaned_population = []
        # Define the transform-related attributes to remove.
        attributes_to_exclude = {
            'num_atoms',
            'num_nodes_per_graph',
            'edge_index',
            'edge_type',
            'atom_feat',
            'atom_feat_full'
        }
        
        for i in range(len(self)):
            original_data = self[i]
            # Create a new, clean Data object.
            new_data = Data()
            
            # Copy all attributes from the original object, skipping the excluded ones.
            for key, value in original_data:
                if key not in attributes_to_exclude:
                    new_data[key] = value
            
            cleaned_population.append(new_data)
            
        return cleaned_population
    
    @classmethod
    def select_elite(cls, population: List[Data], elite_size: int) -> List[Data]:
        """
        Select elite molecules based on fitness scores.
        
        Args:
            population: Population to select from
            elite_size: Number of elite molecules to select
            
        Returns:
            List of elite Data objects sorted by fitness (best first)
        """
        # Filter out molecules without fitness scores
        valid_population = [data for data in population if getattr(data, 'fitness_score', None) is not None]
        
        if len(valid_population) == 0:
            logger.warning("No molecules with fitness scores found")
            return population[:elite_size]
        
        # Sort by fitness score (higher is better)
        def get_fitness_score(m):
            score = getattr(m, 'fitness_score', float('-inf'))
            if score is None:
                return float('-inf')
            return score.item() if hasattr(score, 'item') else score
        
        sorted_population = sorted(valid_population, key=get_fitness_score, reverse=True)
        
        # Select top elite_size molecules
        elite = sorted_population[:elite_size]
        
        logger.info(f"Selected {len(elite)} elite molecules from {len(valid_population)} valid molecules")
        if elite:
            best_fitness = getattr(elite[0], 'fitness_score', None)
            worst_fitness = getattr(elite[-1], 'fitness_score', None)
            if best_fitness is not None and worst_fitness is not None:
                # Handle both tensor and scalar fitness scores
                best_val = best_fitness.item() if hasattr(best_fitness, 'item') else best_fitness
                worst_val = worst_fitness.item() if hasattr(worst_fitness, 'item') else worst_fitness
                logger.info(f"Elite fitness range: {worst_val:.4f} - {best_val:.4f}")
        
        return elite
    
    @classmethod
    def combine_populations(cls, *populations: List[Data]) -> List[Data]:
        """
        Combine multiple populations, ensuring uniqueness based on SMILES strings.
        
        Args:
            *populations: Variable number of populations to combine
            
        Returns:
            Combined population with unique molecules based on SMILES strings
        """
        combined = []
        seen_smiles = set()
        
        for population in populations:
            for data in population:
                # Use SMILES string for uniqueness check
                smiles = getattr(data, 'smiles', None)
                
                # Skip if SMILES is missing or already seen
                if smiles and smiles not in seen_smiles:
                    combined.append(data)
                    seen_smiles.add(smiles)
        
        logger.info(f"Combined {len(populations)} populations into {len(combined)} unique molecules based on SMILES")
        return combined
    
    @classmethod
    def _filter_for_stability(cls, population: List[Data]) -> List[Data]:
        """
        Filter molecules for stability (no disconnected fragments).
        
        Args:
            population: List of molecule Data objects
            
        Returns:
            List of stable molecules
        """
        stable_population = []
        filtered_count = 0
        
        logger.info(f"Filtering {len(population)} molecules for stability...")
        
        for data in population:
            try:
                # Get SMILES from the data object
                smiles = getattr(data, 'smiles', None)
                if not smiles:
                    filtered_count += 1
                    continue
                
                # Check if molecule is stable (no disconnected fragments)
                if "." in smiles:
                    filtered_count += 1
                    continue
                
                # Molecule passed stability check
                stable_population.append(data)
                
            except Exception as e:
                logger.warning(f"Error filtering molecule: {e}")
                filtered_count += 1
                continue
        
        logger.info(f"Stability filtering results: {len(stable_population)} stable molecules from {len(population)} generated")
        logger.info(f"Filtered out {filtered_count} unstable molecules")
        
        return stable_population
    
    @classmethod
    def _filter_for_scaffold(cls, population: List[Data], scaffold_mol2_path: str, remove_h: bool = True) -> List[Data]:
        """
        Filter molecules for scaffold content (contains scaffold as substructure).
        
        Args:
            population: List of molecule Data objects
            scaffold_mol2_path: Path to scaffold MOL2 file
            remove_h: Whether hydrogens are removed
            
        Returns:
            List of molecules containing the scaffold
        """
        from rdkit import Chem
        
        # Load scaffold molecule for filtering
        scaffold_mol_original = Chem.MolFromMol2File(scaffold_mol2_path, removeHs=remove_h)
        if scaffold_mol_original is None:
            raise ValueError(f"Cannot parse scaffold MOL2 file: {scaffold_mol2_path}")
        
        # Convert scaffold to canonical SMILES representation for better matching
        # This fixes issues with radicals and explicit aromatic atom types from MOL2 files
        from utils.reconstruct import mol2smiles
        
        scaffold_smiles = mol2smiles(scaffold_mol_original)
        if scaffold_smiles is None:
            logger.warning("Failed to convert scaffold to canonical SMILES, using original molecule")
            scaffold_mol = scaffold_mol_original
        else:
            logger.info(f"Original scaffold SMILES (with radicals): {Chem.MolToSmiles(scaffold_mol_original)}")
            
            # Fix problematic SMILES with radicals or charges using centralized function
            from utils.mol2_parser import fix_mol2_radicals_and_charges
            scaffold_smiles = fix_mol2_radicals_and_charges(scaffold_mol_original, scaffold_smiles)
            
            logger.info(f"Canonical scaffold SMILES (radicals removed): {scaffold_smiles}")
            
            # Create molecule from canonical SMILES for better substructure matching
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold_mol is None:
                logger.warning("Failed to create molecule from canonical SMILES, using original")
                scaffold_mol = scaffold_mol_original
            else:
                logger.info("Successfully created standardized scaffold molecule")
        
        scaffold_population = []
        scaffold_filtered_count = 0
        
        logger.info(f"Filtering {len(population)} molecules for scaffold content...")
        
        for data in population:
            try:
                # Get SMILES from the data object
                smiles = getattr(data, 'smiles', None)
                if not smiles:
                    scaffold_filtered_count += 1
                    continue
                
                # Convert SMILES to molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scaffold_filtered_count += 1
                    continue
                
                # Check if molecule contains the scaffold as a substructure
                if mol.HasSubstructMatch(scaffold_mol):
                    scaffold_population.append(data)
                else:
                    scaffold_filtered_count += 1
                
            except Exception as e:
                logger.warning(f"Error filtering molecule for scaffold: {e}")
                scaffold_filtered_count += 1
                continue
        
        logger.info(f"Scaffold filtering results: {len(scaffold_population)} scaffold molecules from {len(population)} input")
        logger.info(f"Filtered out {scaffold_filtered_count} molecules without scaffold")
        
        return scaffold_population
    
    @classmethod
    def _filter_molecules(cls, population: List[Data], scaffold_mol2_path: Optional[str] = None, remove_h: bool = True) -> List[Data]:
        """
        Apply complete filtering pipeline: stability + scaffold (if scaffold mode).
        
        Args:
            population: List of molecule Data objects
            scaffold_mol2_path: Optional path to scaffold MOL2 file
            remove_h: Whether hydrogens are removed
            
        Returns:
            List of filtered molecules
        """
        # Step 1: Filter for stability
        filtered_population = cls._filter_for_stability(population)
        
        # Step 2: Filter for scaffold if in scaffold mode
        if scaffold_mol2_path is not None:
            filtered_population = cls._filter_for_scaffold(filtered_population, scaffold_mol2_path, remove_h)
        
        return filtered_population
    
    @classmethod
    def _select_final_population(cls, population: List[Data], target_size: int) -> List[Data]:
        """
        Select final population size from filtered molecules.
        
        Args:
            population: List of filtered molecule Data objects
            target_size: Target population size
            
        Returns:
            List of selected molecules (either all if fewer than target, or randomly selected if more than target)
        """
        if len(population) >= target_size:
            # Randomly select target number from available molecules
            import random
            random.shuffle(population)
            selected_population = population[:target_size]
            logger.info(f"Randomly selected {len(selected_population)} molecules from {len(population)} filtered molecules")
            return selected_population
        else:
            # Use all available molecules (fewer than target)
            logger.info(f"Using all {len(population)} filtered molecules (target was {target_size})")
            return population
    
    @classmethod
    def initialize_population(cls, 
                            model, 
                            dataset_info, 
                            num_samples: int,
                            batch_size: int = 32,
                            sampling_params: Optional[Dict[str, Any]] = None,
                            context: Optional[List[str]] = None,
                            device: str = 'cuda',
                            model_config: Optional[Any] = None,
                            output_dir: str = './output_ga',
                            pre_transform=None,
                            scaffold_mol2_path: Optional[str] = None,
                            remove_h: bool = True,
                            target_population_size: Optional[int] = None,
                            positioning_strategy: str = 'plane_through_origin') -> 'GAPopulationDataset':
        """
        Initialize population by generating molecules using the model.
        
        Args:
            model: Diffusion model for molecule generation
            dataset_info: Dataset information for molecule reconstruction
            num_samples: Number of molecules to generate (usually scaled up for filtering)
            batch_size: Batch size for generation
            sampling_params: Sampling parameters for the model
            context: Context for conditional generation
            device: Device for tensor operations
            model_config: Model configuration
            output_dir: Output directory for saving population
            scaffold_mol2_path: Path to scaffold MOL2 file for fixed structure generation
            remove_h: Whether to remove hydrogen atoms
            target_population_size: Target final population size after filtering (if None, use all filtered molecules)
            positioning_strategy: Scaffold positioning strategy ('plane_through_origin', 'free_position', 'sphere_constraint', 'center_only')
            
        Returns:
            GAPopulationDataset instance with generated population
        """
        # Check if scaffold-based generation is requested
        if scaffold_mol2_path is not None:
            # Use scaffold-based molecule generation
            from utils.molecule_generation import generate_molecules_with_fixed_structure_to_population
            
            logger.info(f"Initializing population with fixed scaffold: {scaffold_mol2_path}")
            logger.info(f"Using positioning strategy: {positioning_strategy}")
            population = generate_molecules_with_fixed_structure_to_population(
                model=model,
                dataset_info=dataset_info,
                num_samples=num_samples,
                batch_size=batch_size,
                sampling_params=sampling_params,
                mol2_path=scaffold_mol2_path,
                context=context,
                device=device,
                model_config=model_config,
                remove_h=remove_h,
                positioning_strategy=positioning_strategy,
                log_prefix="Scaffold-based population generation"
            )
        else:
            # Use standard molecule generation
            from utils.molecule_generation import generate_molecules_to_population
            
            # Generate molecules using shared logic
            population = generate_molecules_to_population(
                model=model,
                dataset_info=dataset_info,
                num_samples=num_samples,
                batch_size=batch_size,
                sampling_params=sampling_params,
                context=context,
                device=device,
                model_config=model_config,
                log_prefix="Population generation"
            )
        
        # Check if generation was successful
        if len(population) == 0:
            if scaffold_mol2_path is not None:
                raise ValueError(f"Scaffold-based molecule generation failed to produce any valid molecules. "
                               f"Check scaffold file: {scaffold_mol2_path}")
            else:
                raise ValueError("Standard molecule generation failed to produce any valid molecules. "
                               "Check model, sampling parameters, or generation settings.")
        
        logger.info(f"Generated {len(population)} molecules successfully")
        
        # Apply filtering pipeline
        population = cls._filter_molecules(population, scaffold_mol2_path, remove_h)
        
        if len(population) == 0:
            if scaffold_mol2_path is not None:
                raise ValueError("All generated molecules were filtered out (no scaffold or instability). "
                               "Check scaffold file, generation parameters, or model quality.")
            else:
                raise ValueError("All generated molecules were filtered out due to instability. "
                               "Check generation parameters or model quality.")
        
        # Apply population size selection logic
        if target_population_size is not None:
            population = cls._select_final_population(population, target_population_size)
        
        # Create initial directory
        
        # Create dataset from generated population (without filepath to avoid automatic saving)
        dataset = cls.from_population_list_to_class(population, pre_transform=pre_transform)
        
        
        logger.info(f"Initialized population of {len(population)} molecules")

        return dataset


def create_data_loader(population_data: List[Data], 
                      batch_size: int = 32, 
                      shuffle: bool = True,
                      transform=None,
                      pre_transform=None) -> DataLoader:
    """
    Create a DataLoader from population data.
    
    Args:
        population_data: List of Data objects
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        transform: Transform to apply on-the-fly
        pre_transform: Transform to apply during dataset creation
        
    Returns:
        DataLoader instance
    """
    dataset = GAPopulationDataset(
        population_data=population_data, 
        transform=transform,
        pre_transform=pre_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 