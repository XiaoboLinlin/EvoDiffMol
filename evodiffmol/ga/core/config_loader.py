"""
Configuration loader for genetic algorithm settings.
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import GAConfig


class GAConfigLoader:
    """Loader for GA configuration files."""
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file (.json or .yml/.yaml)
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                # Try YAML first, then JSON
                content = f.read()
                f.seek(0)
                try:
                    return yaml.safe_load(content)
                except yaml.YAMLError:
                    return json.loads(content)
    
    @staticmethod
    def load_from_json(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file (deprecated - use load_from_file).
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        return GAConfigLoader.load_from_file(config_path)
    
    @staticmethod
    def create_ga_config(config_dict: Dict[str, Any], output_dir: Optional[str] = None) -> GAConfig:
        """Create GAConfig from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            output_dir: Override output directory
            
        Returns:
            GAConfig object
        """
        ga_config_dict = config_dict.get('genetic_algorithm', {})
        scoring_config = config_dict.get('scoring_operator', {})
        output_config = config_dict.get('output_config', {})
        sampling_config = config_dict.get('sampling_parameters', {})
        scaffold_config = config_dict.get('scaffold', {})
        
        # No need to extract weights when using harmonic mean
        fitness_weights = {}
        
        # Use output_dir override if provided
        final_output_dir = output_dir or output_config.get('output_dir', './ga_output')
        
        return GAConfig(
            elite_size=ga_config_dict.get('elite_size', 1000),
            ga_epochs=ga_config_dict.get('ga_epochs', 50),
            adaptive=ga_config_dict.get('adaptive', True),
            train_epochs_per_ga=ga_config_dict.get('train_epochs_per_ga', 1),
            batch_size=ga_config_dict.get('batch_size', 32),
            generation_batch_size=ga_config_dict.get('batch_size', 32),  # Uses same batch_size parameter
            max_generation_attempts=3,
            num_scale_factor=ga_config_dict.get('num_scale_factor', 1.0),
            scaffold_scale_factor=scaffold_config.get('scale_factor'),  # Can be None
            positioning_strategy=scaffold_config.get('positioning_strategy', 'plane_through_origin'),  # Default to plane_through_origin
            sampling_parameters=sampling_config if sampling_config else None,  # Will use defaults from __post_init__
            fitness_weights=fitness_weights,
            use_harmonic_mean=ga_config_dict.get('use_harmonic_mean', True),
            output_dir=final_output_dir,
            checkpoint_freq=ga_config_dict.get('checkpoint_freq', -1)
        )
    
    @staticmethod
    def get_scoring_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scoring configuration from config dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Scoring configuration
        """
        return config_dict.get('scoring_operator', {})
    
    @staticmethod
    def find_config_files(config_dir: str = 'ga_config') -> List[str]:
        """Find all GA configuration files in directory.
        
        Args:
            config_dir: Directory to search for config files
            
        Returns:
            List of configuration file paths
        """
        if not os.path.exists(config_dir):
            return []
        
        config_files = []
        for file in os.listdir(config_dir):
            if file.endswith(('.json', '.yml', '.yaml')):
                config_files.append(os.path.join(config_dir, file))
        
        return sorted(config_files)
    
    @staticmethod
    def list_available_configs(config_dir: str = 'ga_config') -> None:
        """Print available configuration files.
        
        Args:
            config_dir: Directory to search for config files
        """
        config_files = GAConfigLoader.find_config_files(config_dir)
        
        if not config_files:
            print(f"No configuration files found in {config_dir}")
            return
        
        print(f"Available GA configurations in {config_dir}:")
        for config_file in config_files:
            config_name = Path(config_file).stem
            print(f"  - {config_name}: {config_file}")
            
            # Try to load and show brief description
            try:
                config_dict = GAConfigLoader.load_from_file(config_file)
                scoring_config = config_dict.get('scoring_operator', {})
                scoring_names = scoring_config.get('scoring_names', [])
                if scoring_names:
                    print(f"    Properties: {', '.join(scoring_names)}")
            except Exception as e:
                print(f"    (Error loading config: {e})")
        print() 