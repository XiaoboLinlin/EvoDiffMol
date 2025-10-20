"""
Main genetic trainer that orchestrates the genetic algorithm process.
"""

import os
import logging
import time
import shutil
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch_geometric.data
from torch_geometric.data import DataLoader, Data
from tqdm.auto import tqdm

from .base import AbstractTrainer, GAConfig
from ..population.population_dataset import GAPopulationDataset, create_data_loader
from utils.mol2_writer import write_mol2_structure
from utils.reconstruct import build_molecule

# Import scoring components (absolute import)
from scoring.scoring import MolecularScoring, smiles_to_mol

logger = logging.getLogger(__name__)


class GeneticTrainer(AbstractTrainer):
    """
    Main trainer for genetic algorithm with adaptive diffusion model training.
    
    This class orchestrates the entire genetic algorithm process including:
    - Population initialization and management
    - Fitness evaluation with harmonic mean
    - Adaptive model training on elite populations
    - Molecule generation and selection
    """
    
    def __init__(
        self,
        config: GAConfig,
        model: Any,
        model_config: Any,
        optimizer_global: Any,
        optimizer_local: Any,
        dataset: Any,
        dataset_info: Dict,
        transforms: Any,
        device: str = 'cuda',
        context: Optional[List[str]] = None,
        scoring_config: Optional[Dict[str, Any]] = None,
        scaffold_mol2_path: Optional[str] = None,
        remove_h: bool = True,
        blend_mode: bool = False,
        blend_pattern: Optional[str] = None
    ):
        """
        Initialize genetic trainer.
        
        Args:
            config: Genetic algorithm configuration
            model: Diffusion model for molecule generation
            model_config: Model configuration with sampling parameters
            optimizer_global: Global optimizer
            optimizer_local: Local optimizer  
            dataset: Training dataset
            dataset_info: Dataset information for molecule reconstruction
            transforms: PyTorch Geometric transforms
            device: Computing device
            context: Context properties for conditional generation
            scoring_config: Configuration for molecular scoring
        """
        self.config = config
        self.model = model
        self.model_config = model_config
        self.optimizer_global = optimizer_global
        self.optimizer_local = optimizer_local
        self.dataset = dataset
        self.dataset_info = dataset_info
        self.transforms = transforms
        self.device = device
        self.context = context or []
        
        # Store components for population management
        self.model = model
        self.model_config = model_config
        self.dataset_info = dataset_info
        self.context = context
        self.device = device
        
        # Scaffold-related attributes
        self.scaffold_mol2_path = scaffold_mol2_path
        self.remove_h = remove_h
        self.scaffold_mode = scaffold_mol2_path is not None
        
        # Blend mode attributes
        self.blend_mode = blend_mode or (blend_pattern is not None)  # Auto-enable if pattern provided
        self.blend_pattern = blend_pattern
        self.current_blend_phase = 'fixed'  # Start with fixed phase
        self.blend_epoch_counter = 0
        self.blend_schedule = self._parse_blend_pattern(blend_pattern) if self.blend_mode else None
        
        # Initialize molecular scoring
        self._setup_scoring(scoring_config)
        
        # Training state
        self.current_epoch = 0
        self.best_fitness = 0.0
        self.training_history = []
        
        # Setup logging
        self.log_dir = Path(config.output_dir) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GeneticTrainer with config: {config}")
        if self.blend_mode:
            logger.info(f"Blend mode enabled with pattern: {self.blend_pattern}")
            logger.info(f"Blend schedule: {self.blend_schedule}")
    
    def _parse_blend_pattern(self, pattern: str) -> Dict[str, int]:
        """Parse blend pattern string into schedule.
        
        Args:
            pattern: Pattern like 'f2-a2' (2 fixed epochs, 2 adaptive epochs)
            
        Returns:
            Dictionary with 'fixed' and 'adaptive' epoch counts
        """
        if not pattern:
            return {'fixed': 2, 'adaptive': 2}  # Default
            
        try:
            parts = pattern.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid pattern format: {pattern}")
                
            fixed_part = parts[0].strip()
            adaptive_part = parts[1].strip()
            
            # Parse fixed epochs (e.g., 'f2' -> 2)
            if fixed_part.startswith('f'):
                fixed_epochs = int(fixed_part[1:])
            else:
                raise ValueError(f"Fixed part must start with 'f': {fixed_part}")
                
            # Parse adaptive epochs (e.g., 'a2' -> 2)  
            if adaptive_part.startswith('a'):
                adaptive_epochs = int(adaptive_part[1:])
            else:
                raise ValueError(f"Adaptive part must start with 'a': {adaptive_part}")
                
            return {'fixed': fixed_epochs, 'adaptive': adaptive_epochs}
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse blend pattern '{pattern}': {e}. Using default f2-a2")
            return {'fixed': 2, 'adaptive': 2}
    
    def _determine_blend_mode_for_epoch(self, epoch: int) -> bool:
        """Determine if current epoch should use adaptive mode based on blend pattern.
        
        Args:
            epoch: Current epoch number (0-based)
            
        Returns:
            True if should use adaptive mode, False for fixed mode
        """
        if not self.blend_mode or not self.blend_schedule:
            return self.config.adaptive
            
        # Calculate total cycle length
        fixed_epochs = self.blend_schedule['fixed']
        adaptive_epochs = self.blend_schedule['adaptive']
        cycle_length = fixed_epochs + adaptive_epochs
        
        # Determine position within current cycle
        cycle_position = epoch % cycle_length
        
        # First part of cycle is fixed, second part is adaptive
        if cycle_position < fixed_epochs:
            return False  # Fixed mode
        else:
            return True   # Adaptive mode
    
    def _setup_scoring(self, scoring_config: Optional[Dict[str, Any]]) -> None:
        """Setup molecular scoring system.
        
        Args:
            scoring_config: Configuration for molecular scoring
        """
        if scoring_config:
            # Use configuration from JSON file
            scoring_names = scoring_config.get('scoring_names', ['logp', 'qed', 'sa'])
            selection_names = scoring_config.get('selection_names', scoring_names)
            property_config = scoring_config.get('property_config', {})
            scoring_parameters = scoring_config.get('scoring_parameters', {})
            
            # Set fitness function based on config
            fitness_function = None
            if self.config.use_harmonic_mean:
                import scipy.stats
                fitness_function = scipy.stats.hmean
            
            self.molecular_scoring = MolecularScoring(
                scoring_names=scoring_names,
                selection_names=selection_names,
                property_config=property_config,
                scoring_parameters=scoring_parameters,
                fitness_function=fitness_function
            )
            
            logger.info(f"Initialized molecular scoring with properties: {scoring_names}")
        else:
            # Default configuration (fallback)
            import scipy.stats
            self.molecular_scoring = MolecularScoring(
                scoring_names=['logp', 'qed', 'sa'],
                selection_names=['logp', 'qed', 'sa'],
                fitness_function=scipy.stats.hmean if self.config.use_harmonic_mean else np.mean
            )
            logger.info("Using default molecular scoring configuration")
    
        # Molecular scoring is ready to use (no fitting needed)
        logger.info("Molecular scoring initialized and ready")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete genetic algorithm.
        
        Returns:
            Dictionary with final results and statistics
        """
        logger.info("Starting genetic algorithm evolution")
        total_start_time = time.time()  # Record total time from beginning
        
        
        initial_dir = os.path.join(self.config.output_dir, 'initial')
        os.makedirs(initial_dir, exist_ok=True)
        population_path = os.path.join(initial_dir, 'initial_population.pt')
        
        if os.path.exists(population_path):
            # Load existing population
            self.elite_dataset = GAPopulationDataset(filepath=population_path)
            elite_population = self.elite_dataset.to_population_list()
        else:
            # Initialize population
            logger.info("Initializing population...")
            
            # Use consistent initialize_population method for both scaffold and standard modes
            if self.scaffold_mode:
                # Scaffold mode: use scaffold scale factor
                scale_factor = self.config.scaffold_scale_factor if self.config.scaffold_scale_factor is not None else getattr(self.config, 'num_scale_factor', 1.0)
                initial_population_size = int(self.config.elite_size * scale_factor)
                scaffold_path = self.scaffold_mol2_path
                logger.info(f"Initializing scaffold-based population (target elite_size: {self.config.elite_size}, scale_factor: {scale_factor}x)")
            else:
                # Standard mode: use default scale factor
                scale_factor = getattr(self.config, 'num_scale_factor', 1.0)
                initial_population_size = int(self.config.elite_size * scale_factor)
                scaffold_path = None
                logger.info(f"Initializing standard population (target elite_size: {self.config.elite_size}, scale_factor: {scale_factor}x)")
            
            self.elite_dataset = GAPopulationDataset.initialize_population(
                model=self.model,
                dataset_info=self.dataset_info,
                num_samples=initial_population_size,
                batch_size=self.config.generation_batch_size,
                sampling_params=self.config.sampling_parameters,
                context=self.context,
                device=self.device,
                model_config=self.model_config,
                output_dir=self.config.output_dir,
                pre_transform=self.transforms,
                scaffold_mol2_path=scaffold_path,
                remove_h=self.remove_h,
                target_population_size=self.config.elite_size,
                positioning_strategy=self.config.positioning_strategy
            )
            
            final_population_size = len(self.elite_dataset.to_population_list())
            logger.info(f"Successfully initialized population with {final_population_size} molecules")
            self.elite_dataset.save_population(population_path)
            elite_population = self.elite_dataset.to_population_list()
        
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        self.molecular_scoring.evaluate_population(elite_population)
        
        # Save SMILES and MOL2 files for the elite population (after evaluation to include all scores)
        self._save_smile_mol2(elite_population, -1)  # -1 indicates initial population
        
        # Display initial population fitness stats
        logger.info("=== INITIAL POPULATION FITNESS STATS ===")
        self._display_current_fitness_stats(elite_population, -1)  # -1 indicates initial population
        
        # Add initial population to training history (Epoch 0)
        initial_stats = self._log_epoch_results(
            epoch=-1,  # -1 for initial population
            elite_population=elite_population, 
            training_metrics={},  # No training for initial population
            epoch_time=0.0
        )
        
        # Log the start of the first epoch
        self._display_current_fitness_stats(elite_population, epoch=0)
        
        # Main genetic algorithm loop
        epoch = -1  # Initialize epoch for the case when ga_epochs = 0
        for epoch in range(self.config.ga_epochs):
            logger.info(f"=== Genetic Algorithm Epoch {epoch + 1}/{self.config.ga_epochs} ===")
            epoch_start_time = time.time()  # Time for this specific epoch
            
            # Handle blend mode: determine if this epoch should be adaptive or fixed
            current_adaptive_setting = self.config.adaptive
            if self.blend_mode:
                current_adaptive_setting = self._determine_blend_mode_for_epoch(epoch)
                logger.info(f"Blend mode: Epoch {epoch + 1} using {'ADAPTIVE' if current_adaptive_setting else 'FIXED'} strategy")
            
            # Adaptive training on the elite population
            if current_adaptive_setting:
                logger.info(f"Adaptive training on elite population for {self.config.train_epochs_per_ga} epoch(s)...")
                data_loader = create_data_loader(
                    population_data=elite_population,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    pre_transform=self.transforms
                )
                
                # Train for the specified number of epochs
                training_metrics = {}
                for train_epoch_idx in range(self.config.train_epochs_per_ga):
                    if self.config.train_epochs_per_ga > 1:
                        logger.info(f"  Training epoch {train_epoch_idx + 1}/{self.config.train_epochs_per_ga}")
                    epoch_metrics = self.train_epoch(data_loader)
                    
                    # Accumulate metrics (keep the latest values for logging)
                    training_metrics.update(epoch_metrics)
                    
                logger.info(f"Completed {self.config.train_epochs_per_ga} training epoch(s)")
            else:
                training_metrics = {}
            
            # Generate new molecules (target = elite_size, scale factor applied internally)
            target_molecules = self.config.elite_size  # Always target the desired elite size
            logger.info(f"Generating molecules to maintain elite population of {target_molecules}...")
            generated_molecules = self._generate_molecules(target_molecules)
            
            # Combine populations
            combined_population = GAPopulationDataset.combine_populations(
                elite_population, generated_molecules
            )
            
            # Evaluate combined population
            logger.info("Evaluating combined population...")
            self.molecular_scoring.evaluate_population(combined_population)
            
            # Select new elite population
            logger.info("Selecting new elite population...")
            elite_population = GAPopulationDataset.select_elite(
                combined_population, self.config.elite_size
            )
            
            # # Create new elite population dataset
            # self.elite_dataset = GAPopulationDataset.from_population_list_to_class(
            #     elite_population, 
            #     pre_transform=self.transforms
            # )
            
            # # Save SMILES and MOL2 files for the elite population
            # self._save_smile_mol2(elite_population, epoch)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            epoch_stats = self._log_epoch_results(
                epoch, elite_population, training_metrics, epoch_time
            )
            
        # Save checkpoint only at the end (when checkpoint_freq is -1 or at final epoch)
        # Skip checkpointing when ga_epochs = 0 (epoch = -1)
        if self.config.ga_epochs > 0:
            if self.config.checkpoint_freq == -1 and epoch == self.config.ga_epochs - 1:
                self._save_checkpoint(epoch, elite_population, epoch_stats)
            elif self.config.checkpoint_freq > 0 and (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch, elite_population, epoch_stats)
        
        # Final results
        total_time = time.time() - total_start_time  # Total time from beginning of run()
        # Save final elite population and compile results
        self._save_smile_mol2(elite_population, 'last')  # Save final results as CSV with epoch_last
        final_results = self._compile_final_results(elite_population, total_time)
        
        logger.info("Genetic algorithm evolution completed successfully!")
        return final_results
    

    
    def _generate_molecules(self, num_molecules: int) -> List[torch_geometric.data.Data]:
        """
        Generate new molecules using the diffusion model.
        Uses scaffold-based generation if scaffold mode is enabled.
        For scaffold mode, generates extra molecules and filters them to contain the scaffold.
        
        Args:
            num_molecules: Target number of molecules to generate
            
        Returns:
            List of generated molecules
        """
        # Use scale factor to generate more molecules than needed for filtering
        if self.scaffold_mode and self.config.scaffold_scale_factor is not None:
            scale_factor = self.config.scaffold_scale_factor
            scaffold_path = self.scaffold_mol2_path
        else:
            scale_factor = getattr(self.config, 'num_scale_factor', 1.0)
            scaffold_path = None if not self.scaffold_mode else self.scaffold_mol2_path
        
        num_to_generate = int(num_molecules * scale_factor)
        logger.info(f"Generating {num_to_generate} molecules (scale factor: {scale_factor}x) for target {num_molecules}")
        
        # Use the same generation and filtering logic as initialization
        # Note: No target_population_size constraint for evolution - generate as many valid molecules as possible
        temp_dataset = GAPopulationDataset.initialize_population(
            model=self.model,
            dataset_info=self.dataset_info,
            num_samples=num_to_generate,
            batch_size=self.config.generation_batch_size,
            sampling_params=self.config.sampling_parameters,
            context=self.context,
            device=self.device,
            model_config=self.model_config,
            output_dir=self.config.output_dir,
            pre_transform=self.transforms,
            scaffold_mol2_path=scaffold_path,
            remove_h=self.remove_h,
            positioning_strategy=self.config.positioning_strategy
        )
        
        # Get the final selected population
        generated_molecules = temp_dataset.to_population_list()
        
        logger.info(f"Generated {len(generated_molecules)} molecules for evolution")
        return generated_molecules
    

    
    def train_epoch(self, data_loader) -> Dict[str, float]:
        """
        Train the model for one epoch on the provided data.
        
        Args:
            data_loader: DataLoader with training data.
            
        Returns:
            Dictionary with training metrics (e.g., loss)
        """
        # Use shared training function
        from utils.training_utils import train_epoch
        
        training_metrics = train_epoch(
            model=self.model,
            train_loader=data_loader,
            optimizer_global=self.optimizer_global,
            optimizer_local=self.optimizer_local,
            config=self.model_config,
            device=self.device,
            writer=None,
            logger=None,
            iteration=None,
            property_norms=None,
            context_args=self.context,
            ema=None
        )
        return training_metrics
    
    def _log_epoch_results(
        self, 
        epoch: int, 
        elite_population: List[torch_geometric.data.Data], 
        training_metrics: Dict[str, float],
        epoch_time: float
    ) -> Dict[str, Any]:
        """Log results for current epoch."""
        
        # Population statistics
        pop_stats = self.elite_dataset.get_population_statistics(elite_population)
        
        # Fitness statistics
        fitness_scores = [getattr(data, 'fitness_score', None) for data in elite_population]
        fitness_scores = [score for score in fitness_scores if score is not None]
        if fitness_scores:
            # Convert tensors to scalars for numpy operations
            fitness_values = [score.item() if hasattr(score, 'item') else score for score in fitness_scores]
            fitness_stats = {
                'fitness_mean': np.mean(fitness_values),
                'fitness_std': np.std(fitness_values),
                'fitness_max': np.max(fitness_values),
                'fitness_min': np.min(fitness_values)
            }
            
            # Update best fitness
            if fitness_stats['fitness_max'] > self.best_fitness:
                self.best_fitness = fitness_stats['fitness_max']
        else:
            fitness_stats = {}
        
        # Compile epoch statistics
        # For initial population (epoch -1), record as epoch 0
        recorded_epoch = 0 if epoch == -1 else epoch + 1
        
        epoch_stats = {
            'epoch': recorded_epoch,
            'epoch_time': epoch_time,
            'population_stats': pop_stats,
            'fitness_stats': fitness_stats,
            'training_metrics': training_metrics,
            'best_fitness': self.best_fitness
        }
        
        # Log key metrics
        if epoch == -1:
            logger.info(f"Initial population (Epoch 0) evaluated in {epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        if fitness_stats:
            logger.info(f"=== ELITE POPULATION FITNESS STATS ===")
            logger.info(f"Average Fitness Score: {fitness_stats['fitness_mean']:.4f}")
            logger.info(f"Best Fitness Score: {fitness_stats['fitness_max']:.4f}")
            logger.info(f"Fitness Std Dev: {fitness_stats['fitness_std']:.4f}")
            logger.info(f"Fitness Range: {fitness_stats['fitness_min']:.4f} - {fitness_stats['fitness_max']:.4f}")
            logger.info(f"Elite Population Size: {len(elite_population)}")
            logger.info(f"================================")
        if training_metrics:
            avg_loss = training_metrics.get('avg_loss', 0)
            logger.info(f"Training - Loss: {avg_loss:.4f}")
            # Also log other available metrics
            if 'grad_norm' in training_metrics:
                logger.info(f"Training - Grad Norm: {training_metrics['grad_norm']:.4f}")
            if 'lr' in training_metrics:
                logger.info(f"Training - Learning Rate: {training_metrics['lr']:.6f}")
        
        # Store in history
        self.training_history.append(epoch_stats)
        
        return epoch_stats
    
    def _display_current_fitness_stats(self, elite_population: List[torch_geometric.data.Data], epoch: int) -> None:
        """Display current elite population fitness statistics at the start of an epoch."""
        # Calculate fitness statistics
        fitness_scores = [getattr(data, 'fitness_score', None) for data in elite_population]
        fitness_scores = [score for score in fitness_scores if score is not None]
        
        if fitness_scores:
            # Convert tensors to scalars for numpy operations
            fitness_values = [score.item() if hasattr(score, 'item') else score for score in fitness_scores]
            avg_fitness = np.mean(fitness_values)
            max_fitness = np.max(fitness_values)
            min_fitness = np.min(fitness_values)
            std_fitness = np.std(fitness_values)
            
            if epoch == -1:
                # Initial population
                logger.info(f"📊 INITIAL POPULATION FITNESS STATS:")
            else:
                # Current elite population at start of epoch
                logger.info(f"📊 CURRENT ELITE POPULATION FITNESS (Epoch {epoch + 1}):")
            
            logger.info(f"   Average Fitness Score: {avg_fitness:.4f}")
            logger.info(f"   Best Fitness Score: {max_fitness:.4f}")
            logger.info(f"   Worst Fitness Score: {min_fitness:.4f}")
            logger.info(f"   Fitness Std Dev: {std_fitness:.4f}")
            logger.info(f"   Population Size: {len(elite_population)}")
            logger.info(f"   Molecules with valid fitness: {len(fitness_scores)}/{len(elite_population)}")
        else:
            if epoch == -1:
                logger.warning(f"⚠️  No valid fitness scores found in initial population")
            else:
                logger.warning(f"⚠️  No valid fitness scores found in elite population for epoch {epoch + 1}")
    
    def _save_checkpoint(self, epoch: int, elite_population: List[torch_geometric.data.Data], stats: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model checkpoint
        model_path = checkpoint_dir / f'model_epoch_{epoch + 1}.pt'
        self.save_checkpoint(str(model_path), {
            'epoch': epoch,
            'elite_population_size': len(elite_population),
            'best_fitness': self.best_fitness,
            'stats': stats
        })
        
        # Save elite population
        pop_path = checkpoint_dir / f'elite_population_epoch_{epoch + 1}.pt'
        self.elite_dataset.save_population(str(pop_path), elite_population)
        
        logger.info(f"Checkpoint saved at epoch {epoch + 1}")
    
    def save_checkpoint(self, filepath: str, metadata: Dict[str, Any]) -> None:
        """Save model checkpoint with metadata."""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_global_state_dict': self.optimizer_global.state_dict(),
            'optimizer_local_state_dict': self.optimizer_local.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint_data, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint and return metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint_data = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer_global.load_state_dict(checkpoint_data['optimizer_global_state_dict'])
        self.optimizer_local.load_state_dict(checkpoint_data['optimizer_local_state_dict'])
        
        # Restore training state
        if 'training_history' in checkpoint_data:
            self.training_history = checkpoint_data['training_history']
        
        if 'metadata' in checkpoint_data and 'best_fitness' in checkpoint_data['metadata']:
            self.best_fitness = checkpoint_data['metadata']['best_fitness']
        
        logger.info(f"Model checkpoint loaded from {filepath}")
        return checkpoint_data.get('metadata', {})
    
    def _save_smile_mol2(self, elite_population: List[torch_geometric.data.Data], epoch) -> None:
        """Save elite population as SMILES and MOL2 files."""
        # Create output directory
        if epoch == -1:
            # Initial population - save to ga_output/qm40_logp_qed_sa/initial
            output_dir = os.path.join(self.config.output_dir, 'initial')
        elif epoch == 'last':
            # Final epoch - save to ga_output/qm40_logp_qed_sa/epoch_last
            output_dir = os.path.join(self.config.output_dir, 'epoch_last')
        else:
            # Regular epochs
            output_dir = os.path.join(self.config.output_dir, f'epoch_{epoch + 1}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        
        for i, data in enumerate(elite_population):
            if hasattr(data, 'smiles') and data.smiles:
                # Create row for CSV with all available properties
                row = {
                    'molecule_id': i + 1,
                    'smiles': data.smiles
                }
                
                # Add fitness score if available
                if hasattr(data, 'fitness_score') and data.fitness_score is not None:
                    fitness_val = data.fitness_score.item() if hasattr(data.fitness_score, 'item') else data.fitness_score
                    row['fitness_score'] = fitness_val
                
                # Add all scoring properties (logP, QED, SA, etc.)
                if hasattr(self, 'molecular_scoring') and hasattr(self.molecular_scoring, '_scoring_names'):
                    for prop_name in self.molecular_scoring._scoring_names:
                        if hasattr(data, prop_name):
                            prop_val = getattr(data, prop_name)
                            if hasattr(prop_val, 'item'):
                                prop_val = prop_val.item()
                            row[prop_name] = prop_val
                
                csv_data.append(row)
        
        # Save CSV file
        if csv_data:
            csv_path = os.path.join(output_dir, 'elite_molecules.csv')
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"  - CSV: {len(csv_data)} molecules -> {csv_path}")
        
        # Save MOL2 files
        mol2_dir = os.path.join(output_dir, 'mol2_files')
        os.makedirs(mol2_dir, exist_ok=True)
        
        scaffold_molecules = 0  # Count molecules with scaffold information
        
        successful_mol2 = 0
        for i, data in enumerate(elite_population):
            if hasattr(data, 'smiles') and data.smiles and hasattr(data, 'pos') and hasattr(data, 'atom_type'):
                # Build molecule for MOL2 writing
                mol = build_molecule(data.pos, data.atom_type_index, self.dataset_info)
                if mol is not None:
                    mol2_path = os.path.join(mol2_dir, f'elite_{i+1}.mol2')
                    
                    # Extract update_mask if available (for scaffold-based generation)
                    update_mask = getattr(data, 'update_mask', None)
                    if update_mask is not None:
                        scaffold_molecules += 1
                    
                    success = write_mol2_structure(mol, data.pos, data.smiles, mol2_path, update_mask=update_mask)
                    if success:
                        successful_mol2 += 1
        
        # Determine epoch name for logging
        if epoch == -1:
            epoch_name = "initial"
        elif epoch == 'last':
            epoch_name = "last"
        else:
            epoch_name = f"{epoch + 1}"
        
        logger.info(f"Saved elite population files for epoch {epoch_name}:")
        logger.info(f"  - MOL2 files: {successful_mol2}/{len(elite_population)} -> {mol2_dir}")
        if scaffold_molecules > 0:
            logger.info(f"  - Scaffold info: {scaffold_molecules}/{successful_mol2} MOL2 files include fixed/generated atom labels")
    
    def _compile_final_results(self, final_elite: List[torch_geometric.data.Data], total_time: float) -> Dict[str, Any]:
        """Compile final results from genetic algorithm run."""
        
        # Final population statistics
        final_stats = self.elite_dataset.get_population_statistics(final_elite)
        
        # Save final results
        final_results = {
            'total_time': total_time,
            'final_population_size': len(final_elite),
            'best_fitness': self.best_fitness,
            'final_stats': final_stats,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        # Save final elite population
        final_pop_path = Path(self.config.output_dir) / 'final_elite_population.pt'
        self.elite_dataset.save_population(str(final_pop_path), final_elite)
        
        # Save results summary to a human-readable text file
        results_path = Path(self.config.output_dir) / 'final_results.txt'
        with open(results_path, 'w') as f:
            f.write("=== Genetic Algorithm Final Results ===\n\n")
            f.write(f"Total Run Time: {total_time:.2f} seconds\n")
            f.write(f"Final Population Size: {len(final_elite)}\n")
            f.write(f"Best Fitness Achieved: {self.best_fitness:.4f}\n\n")
            
            f.write("--- Final Population Statistics ---\n")
            for key, value in final_stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            f.write("--- Training History Summary ---\n")
            f.write("Note: Epoch 0 is initial population evaluation (no training), so Training Loss = N/A\n")
            f.write("      Training Loss values appear from Epoch 1 onwards (actual GA training iterations)\n\n")
            for i, history in enumerate(self.training_history):
                # Fix: Use 'avg_loss' instead of 'loss' 
                training_metrics = history.get('training_metrics', {})
                loss = training_metrics.get('avg_loss', training_metrics.get('loss', 'N/A'))
                loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
                avg_fitness = history.get('fitness_stats', {}).get('fitness_mean', 'N/A')
                avg_fitness_str = f"{avg_fitness:.4f}" if isinstance(avg_fitness, float) else avg_fitness
                f.write(f"Epoch {history['epoch']}: Average Fitness = {avg_fitness_str}, Training Loss = {loss_str}\n")
            f.write("\n")
            
            f.write("--- Configuration ---\n")
            for key, value in self.config.__dict__.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        logger.info(f"Final results summary saved to {results_path}")
        return final_results 