"""
Molecule Generator API

This module provides a high-level API for molecular generation and optimization
using genetic algorithms with diffusion models.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import tempfile

import torch
import yaml
from easydict import EasyDict
import pandas as pd

# Import core components (from root - configs, datasets)
from configs.datasets_config import get_dataset_info

# Import from evodiffmol package (models, ga, utils, scoring)
from .models.epsnet import get_model
from .utils.common import get_optimizer, get_scheduler
from .utils.datasets import General3D
from .utils.dataset_scaffold import GeneralScaffoldDataset
from .utils.transforms import *
from .utils.misc import *

# Import GA components
from .ga import GAConfig, GAConfigLoader, GeneticTrainer

# Import scoring
from .scoring.scoring import MolecularScoring

# Import scaffold utilities
from .utils.mol2_parser import parse_mol2_fixed_structure
from .utils.scaffold_utils import filter_molecules_with_scaffold


class MoleculeGenerator:
    """
    High-level API for molecular generation and optimization.
    
    This class provides a simple interface for:
    - Loading pre-trained diffusion models
    - Optimizing molecules for target properties using genetic algorithms
    - Scaffold-based molecule generation
    
    Example:
        >>> from evodiffmol import MoleculeGenerator
        >>> from evodiffmol.utils.datasets import General3D
        >>> 
        >>> # Load dataset (for metadata only)
        >>> dataset = General3D('moses', split='valid', remove_h=True)
        >>> 
        >>> # Initialize generator
        >>> gen = MoleculeGenerator(
        ...     checkpoint_path="logs_moses/checkpoints/80.pt",
        ...     model_config="configs/general_without_h.yml",
        ...     ga_config="ga_config/moses_production.yml",
        ...     dataset=dataset
        ... )
        >>> 
        >>> # Optimize molecules
        >>> molecules = gen.optimize(
        ...     target_properties={'logp': 4.0, 'qed': 0.9},
        ...     population_size=100,
        ...     generations=50
        ... )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_config: Optional[str] = None,
        ga_config: Optional[str] = None,
        device: str = 'cuda',
        dataset: Optional[Any] = None,
        verbose: bool = True
    ):
        """
        Initialize the molecule generator.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            model_config: Path to model config (.yml file, e.g., configs/general_without_h.yml)
            ga_config: Path to GA config (.yml file, e.g., ga_config/moses_production.yml)
            device: Device for inference ('cuda' or 'cpu')
            dataset: Cached dataset for initial population sampling (recommended)
            verbose: Print initialization information
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.verbose = verbose
        self.dataset = dataset
        
        # Auto-detect configs if not provided
        if model_config is None:
            model_config = self._auto_detect_model_config()
        if ga_config is None:
            ga_config = self._auto_detect_ga_config()
            
        self.model_config_path = model_config
        self.ga_config_path = ga_config
        
        # Load configurations
        self._load_configs()
        
        # Setup model
        self._setup_model()
        
        if self.verbose:
            self._print_initialization_info()
    
    def _auto_detect_model_config(self) -> str:
        """Auto-detect model config file."""
        candidates = [
            'configs/general_without_h.yml',
            'config/general_without_h.yml',
            'configs/general.yml'
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        warnings.warn("Could not auto-detect model config. Using default: configs/general_without_h.yml")
        return 'configs/general_without_h.yml'
    
    def _auto_detect_ga_config(self) -> str:
        """Auto-detect GA config file."""
        candidates = [
            'ga_config/moses_production.yml',
            'ga_config/moses_production.yaml',
            'ga_config/default.yml'
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        warnings.warn("Could not auto-detect GA config. Using default: ga_config/moses_production.yml")
        return 'ga_config/moses_production.yml'
    
    def _load_configs(self):
        """Load model and GA configurations."""
        # Load model config
        with open(self.model_config_path, 'r') as f:
            self.model_config = EasyDict(yaml.safe_load(f))
        
        # Load GA config  
        ga_config_dict = GAConfigLoader.load_from_file(self.ga_config_path)
        self.base_ga_config = ga_config_dict
        
        # Detect dataset from model config
        self.dataset_name = self.model_config.get('dataset', 'moses')
        self.remove_h = self.model_config.get('remove_h', True)
        
        # Get dataset info
        self.dataset_info = get_dataset_info(self.dataset_name, self.remove_h)
    
    def _setup_model(self):
        """Setup the diffusion model."""
        # Prepare context
        self.context = []
        if hasattr(self.model_config, 'context'):
            self.context = self.model_config.context
        
        # Setup model
        self.model_config.model.context = self.context
        self.model_config.model.num_atom = len(self.dataset_info['atom_decoder']) + 1
        
        self.model = get_model(self.model_config.model).to(self.device)
        
        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            if self.verbose:
                print(f"✓ Loaded checkpoint from {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Setup optimizers (required for GeneticTrainer interface)
        from .utils.common import get_optimizer, get_scheduler
        self.optimizer_global = get_optimizer(self.model_config.train.optimizer, self.model.model_global)
        self.scheduler_global = get_scheduler(self.model_config.train.scheduler, self.optimizer_global)
        self.optimizer_local = get_optimizer(self.model_config.train.optimizer, self.model.model_local)
        self.scheduler_local = get_scheduler(self.model_config.train.scheduler, self.optimizer_local)
        
        # Setup transforms (must include AtomFeat to create atom_feat_full attribute)
        from .utils.transforms import Compose, GetAdj, AtomFeat
        self.transforms = Compose([
            CountNodesPerGraph(), 
            GetAdj(), 
            AtomFeat(self.dataset_info['atom_index'])
        ])
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print("=" * 80)
        print("🧬 EvoDiffMol Generator Initialized")
        print("=" * 80)
        print(f"Model checkpoint: {self.checkpoint_path}")
        print(f"Model config:     {self.model_config_path}")
        print(f"GA config:        {self.ga_config_path}")
        print(f"Dataset:          {self.dataset_name}")
        print(f"Device:           {self.device}")
        print(f"Remove H:         {self.remove_h}")
        print("=" * 80)
    
    def optimize(
        self,
        target_properties: Dict[str, float],
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        scaffold_smiles: Optional[str] = None,
        scaffold_mol2_path: Optional[str] = None,
        scaffold_scale_factor: float = 2.5,
        positioning_strategy: str = 'plane_through_origin',
        fine_tune_epochs: int = 0,
        fitness_weights: Optional[Dict[str, float]] = None,
        output_dir: Optional[str] = None,
        save_results: bool = False,
        verbose: Optional[bool] = None,
        return_dataframe: bool = False,
        **ga_params
    ) -> Union[List[str], pd.DataFrame]:
        """
        Optimize molecules using genetic algorithm.
        
        Args:
            target_properties: Target properties to optimize (e.g., {'logp': 4.0, 'qed': 0.9})
            population_size: Population size for GA (default: from config or 100)
            generations: Number of GA generations (default: from config or 50)
            scaffold_smiles: SMILES string of scaffold to maintain (optional)
            scaffold_mol2_path: Path to MOL2 file for scaffold (optional, alternative to scaffold_smiles)
            scaffold_scale_factor: Scale factor for scaffold generation (default: 2.5)
            positioning_strategy: Scaffold positioning strategy (default: 'plane_through_origin')
            fine_tune_epochs: Fine-tuning epochs for scaffold (default: 0)
            fitness_weights: Weights for each property in fitness calculation 
                           (default: equal weights of 1.0 for all properties)
                           Example: {'logp': 2.0, 'qed': 1.0} gives logp twice the importance
            output_dir: Directory to save results (optional, no files saved if None)
            save_results: Save final results and logs (default: False)
            verbose: Print progress information (default: use instance verbose setting)
            return_dataframe: Return DataFrame instead of list (default: False)
            **ga_params: Additional GA parameters (num_scale_factor, batch_size, etc.)
        
        Returns:
            List of SMILES strings or DataFrame with molecules and properties
        """
        if verbose is None:
            verbose = self.verbose
        
        # Setup logging
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create GA config
        ga_config = self._create_ga_config(
            target_properties=target_properties,
            population_size=population_size,
            generations=generations,
            fitness_weights=fitness_weights,
            output_dir=output_dir,
            scaffold_scale_factor=scaffold_scale_factor,
            positioning_strategy=positioning_strategy,
            **ga_params
        )
        
        # Save GA config to output_dir/config/ga_config_used.yml for reproducibility
        if output_dir:
            self._save_ga_config(ga_config, target_properties, output_dir)
        
        # Handle scaffold mode
        # Support both scaffold_smiles (converts to MOL2) and scaffold_mol2_path (direct MOL2 file)
        if scaffold_smiles and scaffold_mol2_path:
            raise ValueError("Cannot specify both scaffold_smiles and scaffold_mol2_path. Choose one.")
        
        # Determine if we're in scaffold mode and get the dataset
        is_scaffold_mode = scaffold_smiles is not None or scaffold_mol2_path is not None
        
        if is_scaffold_mode:
            if scaffold_smiles:
                # Convert SMILES to MOL2 and create filtered dataset
                if verbose:
                    print(f"🔬 Scaffold mode: filtering dataset for scaffold '{scaffold_smiles}'")
                dataset = self._get_scaffold_dataset(scaffold_smiles, output_dir=output_dir)
            elif scaffold_mol2_path:
                # Use provided MOL2 file directly
                if verbose:
                    print(f"🔬 Scaffold mode: using MOL2 file '{scaffold_mol2_path}'")
                # Import GeneralScaffoldDataset for direct MOL2 usage
                from .utils.dataset_scaffold import GeneralScaffoldDataset
                
                # Save scaffold dataset to output_dir if provided, otherwise use datasets/
                scaffold_root = os.path.join(output_dir, 'scaffold_cache') if output_dir else 'datasets'
                
                dataset = GeneralScaffoldDataset(
                    scaffold_mol2_path=scaffold_mol2_path,
                    dataset_name=self.dataset_name,
                    root=scaffold_root,
                    remove_h=self.remove_h,
                    pre_transform=self.transforms  # ← CRITICAL: Include transforms to create atom_feat_full
                )
        else:
            # Regular mode: use provided dataset or load default
            dataset = self.dataset
            if dataset is None:
                if verbose:
                    print("⚠️  No dataset provided. Loading full dataset (this may be slow)...")
                dataset = General3D(self.dataset_name, 'train', 
                                   pre_transform=self.transforms, 
                                   remove_h=self.remove_h)
        
        # Convert target_properties to property_config with preferred_value
        property_config = self._create_property_config(target_properties)
        
        # Create scoring_config from base config + target properties
        scoring_config = self.base_ga_config.get('scoring_operator', {}).copy()
        scoring_config['property_config'] = property_config
        
        # CRITICAL: Override scoring_names and selection_names with target_properties
        # This ensures fitness is calculated only from target properties, not from base config
        target_prop_names = list(target_properties.keys())
        scoring_config['scoring_names'] = target_prop_names
        scoring_config['selection_names'] = target_prop_names
        
        # Auto-detect ADMET properties from target_properties
        # ADMET properties are those not in the basic property list
        basic_properties = {'logp', 'qed', 'sa', 'tpsa'}
        admet_properties = [prop for prop in target_properties.keys() if prop not in basic_properties]
        
        if admet_properties:
            # Add ADMET properties to scoring config
            scoring_config['scoring_admet_names'] = admet_properties
            
            if verbose:
                print(f"🧬 ADMET properties detected: {admet_properties}")
        
        # Determine the scaffold_mol2_path for GeneticTrainer
        # If scaffold_mol2_path was provided directly, use it
        # If scaffold_smiles was provided, _get_scaffold_dataset creates a temporary MOL2 file
        # and stores it in the dataset's scaffold_mol2_path attribute
        trainer_scaffold_mol2 = None
        if scaffold_mol2_path:
            trainer_scaffold_mol2 = scaffold_mol2_path
        elif scaffold_smiles and hasattr(dataset, 'scaffold_mol2_path'):
            trainer_scaffold_mol2 = dataset.scaffold_mol2_path
        
        # Setup unified logging to output_dir/logs/training.log (for both fine-tuning and GA)
        training_logger = None
        if output_dir:
            import logging
            log_dir = os.path.join(output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            training_log_file = os.path.join(log_dir, 'training.log')
            
            # Setup logger for fine-tuning
            ft_logger = logging.getLogger(f'fine_tuning_{id(self)}')
            ft_logger.setLevel(logging.INFO)
            ft_logger.handlers = []
            
            # Setup logger for GA training
            ga_logger = logging.getLogger('evodiffmol.ga.core.trainer')
            ga_logger.setLevel(logging.INFO)
            ga_logger.handlers = []
            
            # Create shared file handler
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh = logging.FileHandler(training_log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            
            # Add file handler to both loggers
            ft_logger.addHandler(fh)
            ga_logger.addHandler(fh)
            
            # Also add console handler if verbose
            if verbose:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                ft_logger.addHandler(ch)
                ga_logger.addHandler(ch)
            
            training_logger = ft_logger
        
        # Fine-tune model on scaffold dataset (if requested)
        if is_scaffold_mode and fine_tune_epochs > 0:
            if verbose:
                print(f"\n🔧 Fine-tuning model on scaffold dataset for {fine_tune_epochs} epochs...")
            
            self._fine_tune_on_scaffold(
                dataset=dataset,
                fine_tune_epochs=fine_tune_epochs,
                batch_size=ga_config.batch_size,
                verbose=verbose,
                output_dir=output_dir,  # Pass output_dir for logging
                logger=training_logger  # Pass the unified logger
            )
            
            if verbose:
                print("✅ Fine-tuning completed!\n")
        
        # Create genetic trainer
        trainer = GeneticTrainer(
            config=ga_config,
            model=self.model,
            model_config=self.model_config,
            optimizer_global=self.optimizer_global,
            optimizer_local=self.optimizer_local,
            dataset=dataset,
            dataset_info=self.dataset_info,
            transforms=self.transforms,
            device=self.device,
            context=self.context,
            scoring_config=scoring_config,
            scaffold_mol2_path=trainer_scaffold_mol2,
            remove_h=self.remove_h
        )
        
        # Run optimization
        if verbose:
            print(f"\n🚀 Starting optimization...")
            print(f"   Target properties: {target_properties}")
            print(f"   Population size:   {ga_config.elite_size}")
            print(f"   Generations:       {ga_config.ga_epochs}")
            if scaffold_smiles:
                print(f"   Scaffold:          {scaffold_smiles}")
            elif scaffold_mol2_path:
                print(f"   Scaffold:          {scaffold_mol2_path}")
        
        # Run optimization
        final_results = trainer.run()
        
        # Store output_dir for DataFrame conversion
        self._last_output_dir = output_dir or (trainer.config.output_dir if hasattr(trainer, 'config') else None)
        
        # Get final results
        final_molecules = self._extract_final_molecules(trainer, output_dir, final_results)
        
        # Clean up temporary scaffold file (only if it was auto-generated from SMILES)
        if scaffold_smiles and trainer_scaffold_mol2 and trainer_scaffold_mol2.startswith('/tmp'):
            try:
                os.remove(trainer_scaffold_mol2)
            except:
                pass
        
        # Return results
        if return_dataframe:
            return self._molecules_to_dataframe(final_molecules, target_properties)
        else:
            return final_molecules
    
    def _create_ga_config(
        self,
        target_properties: Dict[str, float],
        population_size: Optional[int],
        generations: Optional[int],
        fitness_weights: Optional[Dict[str, float]],
        output_dir: Optional[str],
        scaffold_scale_factor: float,
        positioning_strategy: str,
        **ga_params
    ) -> GAConfig:
        """Create GA configuration from parameters.
        
        Converts target_properties into proper property_config with preferred_value.
        fitness_weights defaults to equal weights (1.0) for all properties if not specified.
        """
        # Extract nested sections from base config
        ga_config_dict = self.base_ga_config.get('genetic_algorithm', {})
        sampling_config = self.base_ga_config.get('sampling_parameters', {})
        scaffold_config = self.base_ga_config.get('scaffold', {})
        
        # CRITICAL FIX: fitness_weights should default to equal weights (1.0) for all properties
        # target_properties contains TARGET VALUES (e.g., {'BBB_Martins': 0.0} means target=0)
        # fitness_weights contains IMPORTANCE (e.g., {'BBB_Martins': 1.0} means equal importance)
        if fitness_weights is None:
            # Default: equal weights for all target properties
            fitness_weights = {prop: 1.0 for prop in target_properties.keys()}
        
        # Start with base config, using nested structure
        config_dict = {
            'elite_size': population_size if population_size else ga_config_dict.get('elite_size', 100),
            'ga_epochs': generations if generations is not None else ga_config_dict.get('ga_epochs', 50),
            'batch_size': ga_params.get('batch_size', ga_config_dict.get('batch_size', 32)),
            'num_scale_factor': ga_params.get('num_scale_factor', ga_config_dict.get('num_scale_factor', 2.0)),
            'adaptive': ga_config_dict.get('adaptive', True),
            'train_epochs_per_ga': ga_config_dict.get('train_epochs_per_ga', 1),
            'use_harmonic_mean': ga_config_dict.get('use_harmonic_mean', True),
            'checkpoint_freq': ga_config_dict.get('checkpoint_freq', -1),
            'sampling_parameters': sampling_config if sampling_config else None,
            'output_dir': output_dir if output_dir else './ga_output_temp',
            'scaffold_scale_factor': scaffold_scale_factor,
            'positioning_strategy': positioning_strategy,
            'fitness_weights': fitness_weights,
        }
        
        # Override with any additional parameters
        config_dict.update(ga_params)
        
        # Create GAConfig
        return GAConfig(**config_dict)
    
    def _create_property_config(self, target_properties: Dict[str, float]) -> Dict[str, Dict]:
        """Convert target_properties to property_config with preferred_value.
        
        Also includes configurations for all standard scoring properties (logp, qed, sa, tpsa)
        even if they're not in target_properties, since they're always calculated.
        
        Args:
            target_properties: Dict like {'logp': 2.0, 'qed': 1.0, 'sa': 1.0}
            
        Returns:
            property_config dict with preferred_value for each property
        """
        # Get base property config from GA config
        base_config = self.base_ga_config.get('scoring_operator', {}).get('property_config', {})
        
        # Import comprehensive property configs for ADMET and molecular properties
        try:
            from evodiffmol.scoring.property_configs import get_all_property_configs
            comprehensive_configs = get_all_property_configs()
        except ImportError:
            comprehensive_configs = {}
        
        # Always include standard scoring properties (needed for normalization)
        standard_properties = {'logp', 'qed', 'sa', 'tpsa'}
        all_properties_needed = set(target_properties.keys()) | standard_properties
        
        # Create new property config with preferred_value from target_properties
        property_config = {}
        for prop_name in all_properties_needed:
            target_value = target_properties.get(prop_name)  # None if not a target
            
            if prop_name in base_config:
                # Copy base config for this property
                property_config[prop_name] = base_config[prop_name].copy()
                # Override with preferred_value from target if specified
                if target_value is not None:
                    property_config[prop_name]['preferred_value'] = target_value
                # Remove preferred_range if it exists (use preferred_value instead)
                property_config[prop_name].pop('preferred_range', None)
            elif prop_name in comprehensive_configs:
                # Use comprehensive config (includes ADMET properties)
                property_config[prop_name] = comprehensive_configs[prop_name].copy()
                # Override with preferred_value from target if specified
                if target_value is not None:
                    property_config[prop_name]['preferred_value'] = target_value
            else:
                # Property not found in any configuration - this is an error!
                raise ValueError(
                    f"Property '{prop_name}' not found in property configurations. "
                    f"Available properties: {list(set(list(base_config.keys()) + list(comprehensive_configs.keys())))[:20]}..."
                )
        
        return property_config
    
    def _save_ga_config(self, ga_config: GAConfig, target_properties: Dict[str, float], output_dir: str) -> None:
        """Save GA configuration to YAML file for reproducibility.
        
        Args:
            ga_config: GAConfig object to save
            target_properties: Target properties dict
            output_dir: Output directory
        """
        import yaml
        from datetime import datetime
        
        # Create config directory
        config_dir = os.path.join(output_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'ga_config_used.yml')
        
        # Build the full config dictionary matching the structure
        config_dict = {
            'genetic_algorithm': {
                'elite_size': ga_config.elite_size,
                'batch_size': ga_config.batch_size,
                'ga_epochs': ga_config.ga_epochs,
                'adaptive': ga_config.adaptive,
                'train_epochs_per_ga': ga_config.train_epochs_per_ga,
                'use_harmonic_mean': ga_config.use_harmonic_mean,
                'checkpoint_freq': ga_config.checkpoint_freq,
                'num_scale_factor': ga_config.num_scale_factor,
            },
            'sampling_parameters': ga_config.sampling_parameters if ga_config.sampling_parameters else {},
            'scoring_operator': {
                'scoring_names': list(target_properties.keys()),
                'selection_names': list(target_properties.keys()),
                'property_config': {}
            },
            'output_config': {
                'output_dir': ga_config.output_dir
            }
        }
        
        # Add scaffold config if present
        if ga_config.scaffold_scale_factor is not None:
            config_dict['scaffold'] = {
                'scale_factor': ga_config.scaffold_scale_factor,
                'positioning_strategy': ga_config.positioning_strategy
            }
        
        # Build property_config from target_properties
        for prop_name, target_value in target_properties.items():
            prop_range = self._get_property_range(prop_name)
            config_dict['scoring_operator']['property_config'][prop_name] = {
                'range': prop_range,
                'preferred_value': float(target_value)
            }
            
            # Add higher_is_better/lower_is_better flags
            if prop_name in ['sa']:
                config_dict['scoring_operator']['property_config'][prop_name]['higher_is_better'] = False
            elif prop_name in ['qed']:
                config_dict['scoring_operator']['property_config'][prop_name]['higher_is_better'] = True
        
        # Write config with header
        with open(config_file, 'w') as f:
            f.write(f"# Generated GA Configuration\n")
            f.write(f"# Target properties: {', '.join([f'{k}={v}' for k, v in target_properties.items()])}\n")
            f.write(f"# Generated on: {datetime.now()}\n")
            f.write(f"# Training Configuration:\n")
            f.write(f"#   Elite size: {ga_config.elite_size}\n")
            f.write(f"#   Batch size: {ga_config.batch_size}\n")
            f.write(f"#   GA epochs: {ga_config.ga_epochs}\n")
            f.write(f"#   Train epochs per GA: {ga_config.train_epochs_per_ga}\n")
            f.write(f"#   Scale factor: {ga_config.num_scale_factor} (generates {int(ga_config.elite_size * ga_config.num_scale_factor)} molecules per generation)\n\n")
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def _get_property_range(self, prop: str) -> List[float]:
        """Get valid range for each property.
        
        Args:
            prop: Property name
            
        Returns:
            List of [min, max] values for the property
        """
        ranges = {
            'logp': [-4.5, 6],
            'tpsa': [0, 188],
            'sa': [1, 10],
            'qed': [0, 1]
        }
        return ranges.get(prop, [0, 1])
    
    def _create_scaffold_mol2(self, scaffold_smiles: str) -> str:
        """Create temporary MOL2 file from SMILES."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile
        
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {scaffold_smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        
        # Write to temporary MOL2 file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mol2', mode='w')
        mol2_content = Chem.MolToMolBlock(mol)
        
        # Convert MOL block to MOL2 format (simplified)
        # Note: For production, use a proper MOL2 writer
        temp_file.write(mol2_content)
        temp_file.close()
        
        return temp_file.name
    
    def _get_scaffold_dataset(self, scaffold_smiles: str, output_dir: Optional[str] = None):
        """Get or create scaffold-filtered dataset."""
        # Import the helper function
        from .utils.dataset_scaffold_smiles import create_scaffold_dataset_from_smiles
        
        # Determine root directory: use output_dir/scaffold_cache if provided, otherwise 'datasets'
        scaffold_root = os.path.join(output_dir, 'scaffold_cache') if output_dir else 'datasets'
        
        # Create scaffold dataset with pre_transform (CRITICAL for fine-tuning compatibility)
        dataset = create_scaffold_dataset_from_smiles(
            scaffold_smiles=scaffold_smiles,
            dataset_name=self.dataset_name,
            remove_h=self.remove_h,
            root=scaffold_root,
            pre_transform=self.transforms  # ← FIX: Include transforms to create atom_feat_full
        )
        
        return dataset
    
    def _extract_final_molecules(self, trainer: GeneticTrainer, output_dir: Optional[str], final_results: Dict[str, Any]) -> List[str]:
        """Extract final molecules from trainer.
        
        Returns list of SMILES strings.
        """
        # Method 1: Try to load from epoch_last/elite_molecules.csv if available (top level)
        if output_dir:
            csv_path = os.path.join(output_dir, 'epoch_last', 'elite_molecules.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'smiles' in df.columns:
                    return df['smiles'].tolist()
                elif 'SMILES' in df.columns:
                    return df['SMILES'].tolist()
        
        # Method 2: Extract from trainer's elite_dataset
        if hasattr(trainer, 'elite_dataset'):
            try:
                population_list = trainer.elite_dataset.to_population_list()
                smiles_list = []
                for data in population_list:
                    if hasattr(data, 'smile'):
                        smiles_list.append(data.smile)
                    elif hasattr(data, 'smiles'):
                        smiles_list.append(data.smiles)
                if smiles_list:
                    return smiles_list
            except Exception as e:
                warnings.warn(f"Could not extract from elite_dataset: {e}")
        
        # Method 3: Check if saved to temp output dir
        # When output_dir is None, trainer creates its own temp output
        if hasattr(trainer, 'config') and hasattr(trainer.config, 'output_dir'):
            csv_path = os.path.join(trainer.config.output_dir, 'epoch_last', 'elite_molecules.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'smiles' in df.columns:
                    return df['smiles'].tolist()
                elif 'SMILES' in df.columns:
                    return df['SMILES'].tolist()
        
        # Last resort: return empty list
        warnings.warn("Could not extract final molecules from trainer")
        return []
    
    def _molecules_to_dataframe(self, molecules: List[str], target_properties: Dict[str, float]) -> pd.DataFrame:
        """Convert molecule list to DataFrame with properties.
        
        Loads properties from the saved CSV file which already has all properties calculated.
        
        Args:
            molecules: List of SMILES strings
            target_properties: Dict of properties to include
            
        Returns:
            DataFrame with SMILES and calculated properties
        """
        # Load from saved CSV file which already has all properties
        if hasattr(self, '_last_output_dir') and self._last_output_dir:
            csv_path = os.path.join(self._last_output_dir, 'epoch_last', 'elite_molecules.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Return SMILES + requested properties + fitness_score
                columns_to_keep = ['smiles']
                if 'fitness_score' in df.columns:
                    columns_to_keep.append('fitness_score')
                columns_to_keep.extend([p for p in target_properties.keys() if p in df.columns])
                
                if all(col in df.columns for col in columns_to_keep):
                    return df[columns_to_keep]
        
        # If we reach here, something went wrong - just return basic DataFrame
        warnings.warn("Could not load properties from CSV, returning SMILES only")
        return pd.DataFrame({'smiles': molecules})
    
    def _fine_tune_on_scaffold(self, dataset, fine_tune_epochs: int, batch_size: int, verbose: bool, output_dir: str = None, logger=None):
        """Fine-tune the model on scaffold-filtered dataset with caching.
        
        Fine-tuned model is saved to output_dir/fine_tuned_models/scaffold_fine_tuned_model.pt
        and reused on subsequent runs in the same output_dir. Logs are saved to output_dir/logs/training.log.
        
        Args:
            dataset: Scaffold-filtered dataset (train split)
            fine_tune_epochs: Number of epochs to fine-tune
            batch_size: Batch size for training
            verbose: Print progress
            output_dir: Output directory for logs and model cache
            logger: Logger instance for logging (optional)
        """
        from torch_geometric.data import DataLoader
        from .utils.training_utils import train_epoch, validate_epoch
        import torch
        
        # Determine fine-tuned model path - save directly to output_dir/fine_tuned_models/
        # No scaffold ID needed since output_dir is already unique
        scaffold_mol2 = getattr(dataset, 'scaffold_mol2_path', None)
        
        if output_dir:
            # Store fine-tuned model directly in output_dir
            fine_tune_dir = os.path.join(output_dir, 'fine_tuned_models')
            os.makedirs(fine_tune_dir, exist_ok=True)
            fine_tune_checkpoint = os.path.join(fine_tune_dir, 'scaffold_fine_tuned_model.pt')
            
            if logger:
                logger.info(f"Scaffold MOL2: {scaffold_mol2}")
                logger.info(f"Model will be saved to: {fine_tune_checkpoint}")
        else:
            # Fallback: no caching without output_dir
            fine_tune_checkpoint = None
            if verbose:
                print(f"   ⚠️  No output_dir provided, fine-tuning without cache")
            if logger:
                logger.warning("No output_dir provided, fine-tuning without cache")
        
        # Check if fine-tuned model already exists
        if fine_tune_checkpoint and os.path.exists(fine_tune_checkpoint):
            if verbose:
                print(f"   ✓ Found cached fine-tuned model!")
                print(f"   Loading from: {fine_tune_checkpoint}")
            if logger:
                logger.info("Found cached fine-tuned model")
                logger.info(f"Loading from: {fine_tune_checkpoint}")
            
            # Load existing fine-tuned model
            ckpt = torch.load(fine_tune_checkpoint, weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            
            # Optionally load optimizer states
            if 'optimizer_global' in ckpt and self.optimizer_global:
                self.optimizer_global.load_state_dict(ckpt['optimizer_global'])
            if 'optimizer_local' in ckpt and self.optimizer_local:
                self.optimizer_local.load_state_dict(ckpt['optimizer_local'])
            
            best_val_loss = ckpt.get('best_val_loss', 'N/A')
            if verbose:
                print(f"   Best validation loss: {best_val_loss}")
                print(f"   Skipping fine-tuning (using cached model)")
            if logger:
                logger.info(f"Best validation loss: {best_val_loss}")
                logger.info("Skipping fine-tuning (using cached model)")
            
            return
        
        # No cached model found - need to fine-tune
        if verbose:
            print(f"   Training on {len(dataset)} scaffold molecules...")
        if logger:
            logger.info(f"Training on {len(dataset)} scaffold molecules")
            logger.info(f"Fine-tuning for {fine_tune_epochs} epochs")
        
        # Load validation dataset from same scaffold root as training dataset
        from .utils.dataset_scaffold import GeneralScaffoldDataset
        scaffold_mol2 = getattr(dataset, 'scaffold_mol2_path', None)
        scaffold_root = getattr(dataset, 'root', None)
        
        val_dataset = None
        if scaffold_mol2 and scaffold_root:
            try:
                # Use SAME root as training dataset (could be output_dir/scaffold_cache or datasets/)
                val_dataset = GeneralScaffoldDataset(
                    scaffold_mol2_path=scaffold_mol2,
                    dataset_name=self.dataset_name,
                    root=scaffold_root,  # Same root as train dataset
                    split='valid',
                    min_molecules=50,
                    max_molecules=2500,
                    remove_h=self.remove_h,
                    pre_transform=self.transforms,
                    project_name=None  # Important: No project-specific naming
                )
                if verbose:
                    print(f"   Validation set: {len(val_dataset)} molecules")
                if logger:
                    logger.info(f"Validation set: {len(val_dataset)} molecules")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Could not load validation set: {e}")
                if logger:
                    logger.warning(f"Could not load validation set: {e}")
                val_dataset = None
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size // 2, shuffle=False) if val_dataset else None
        
        # Fine-tune the model
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(fine_tune_epochs):
            if verbose:
                print(f"   Epoch {epoch+1}/{fine_tune_epochs}...", end=' ')
            if logger:
                logger.info(f"Epoch {epoch+1}/{fine_tune_epochs}")
            
            # Train one epoch
            train_metrics = train_epoch(
                model=self.model,
                train_loader=train_loader,
                optimizer_global=self.optimizer_global,
                optimizer_local=self.optimizer_local,
                config=self.model_config,
                device=self.device,
                writer=None,
                logger=None,
                iteration=epoch,
                property_norms=None,
                context_args=[],
                ema=None
            )
            train_loss = train_metrics['avg_loss']  # Extract scalar from dict
            
            # Validate if validation set available
            if val_loader:
                val_loss = validate_epoch(
                    model=self.model,
                    val_loader=val_loader,
                    config=self.model_config,
                    device=self.device,
                    scheduler_global=None,
                    scheduler_local=None,
                    writer=None,
                    logger=None,
                    iteration=epoch,
                    property_norms=None,
                    context_args=[],
                    ema=None
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if verbose:
                    print(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if logger:
                    logger.info(f"Epoch {epoch+1}/{fine_tune_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val_loss:.4f}")
            else:
                if verbose:
                    print(f"train_loss={train_loss:.4f}")
                if logger:
                    logger.info(f"Epoch {epoch+1}/{fine_tune_epochs}: train_loss={train_loss:.4f}")
        
        self.model.eval()
        
        # Save fine-tuned model for future use
        if fine_tune_checkpoint:
            if verbose:
                print(f"   💾 Saving fine-tuned model...")
                print(f"   Location: {fine_tune_checkpoint}")
            if logger:
                logger.info("Saving fine-tuned model")
                logger.info(f"Location: {fine_tune_checkpoint}")
            
            os.makedirs(os.path.dirname(fine_tune_checkpoint), exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'optimizer_global': self.optimizer_global.state_dict() if self.optimizer_global else None,
                'optimizer_local': self.optimizer_local.state_dict() if self.optimizer_local else None,
                'best_val_loss': best_val_loss if val_loader else None,
                'config': self.model_config,
                'scaffold_mol2_path': scaffold_mol2,
                'fine_tune_epochs': fine_tune_epochs,
            }, fine_tune_checkpoint)
            
            if verbose:
                print(f"   ✓ Model saved to: {fine_tune_checkpoint}")
            if logger:
                logger.info(f"Model saved successfully to: {fine_tune_checkpoint}")
                logger.info("Future runs in this output_dir will use cached model")


# Alias for convenience
Generator = MoleculeGenerator

