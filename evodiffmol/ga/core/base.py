"""
Base classes and data structures for genetic algorithm.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np
from torch_geometric.data import Data





@dataclass
class GAConfig:
    """
    Configuration parameters for genetic algorithm.
    """
    # Population parameters
    elite_size: int = 1000
    ga_epochs: int = 50
    
    # Training parameters
    adaptive: bool = True
    train_epochs_per_ga: int = 1
    batch_size: int = 32
    
    # Generation parameters
    generation_batch_size: int = 32
    max_generation_attempts: int = 3
    num_scale_factor: float = 1.0  # Scale factor for molecule generation
    
    # Scaffold-specific parameters
    scaffold_scale_factor: Optional[float] = None  # Scale factor for scaffold-based generation
    positioning_strategy: str = 'plane_through_origin'  # Positioning strategy for scaffold placement
    
    # Sampling parameters for molecule generation
    sampling_parameters: Optional[Dict[str, Any]] = None
    
    # Fitness parameters
    fitness_weights: Dict[str, float] = None
    use_harmonic_mean: bool = True
    
    # File paths
    output_dir: str = './ga_output'
    checkpoint_freq: int = float('inf')        # Save full model checkpoints (heavy)
    save_epoch_csv_freq: int = -1              # Save epoch CSV files only (lightweight)
    

    
    def __post_init__(self):
        """Set default fitness weights if not provided."""
        if self.fitness_weights is None:
            self.fitness_weights = {
                'energy': 0.3,
                'drug_likeness': 0.25,
                'synthetic_accessibility': 0.2,
                'novelty': 0.15,
                'diversity': 0.1
            }
        
        # Set default sampling parameters if not provided
        if self.sampling_parameters is None:
            self.sampling_parameters = {
                'n_steps': 1000,  # Number of diffusion steps (standard for diffusion models)
                'step_lr': 1e-6,
                'w_global_pos': 1.0,
                'w_global_node': 4.0,  # Keep original value from test_eval defaults
                'w_local_pos': 1.0,
                'w_local_node': 5.0,   # Keep original value from test_eval defaults  
                'global_start_sigma': float('inf'),
                'clip': 1000.0,
                'clip_local': None,
                'sampling_type': 'generalized',
                'eta': 1.0
            }





class AbstractFitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    @abstractmethod
    def evaluate(self, data: Data) -> float:
        """Evaluate fitness score for a single molecule."""
        pass
    
    @abstractmethod
    def evaluate_batch(self, data_list: List[Data]) -> np.ndarray:
        """Evaluate fitness scores for a batch of molecules."""
        pass
    
    @abstractmethod
    def get_property_scores(self, data: Data) -> Dict[str, float]:
        """Get individual property scores for a molecule."""
        pass


class AbstractMoleculeGenerator(ABC):
    """Abstract base class for molecule generation."""
    
    @abstractmethod
    def generate(self, num_molecules: int, context: Optional[Any] = None) -> List[Data]:
        """Generate new molecules using the diffusion model."""
        pass
    
    @abstractmethod
    def set_model(self, model: Any) -> None:
        """Set the underlying diffusion model."""
        pass


class AbstractTrainer(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train_epoch(self, data_list: List[Data]) -> Dict[str, float]:
        """Train model for one epoch on given molecules."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str, metadata: Dict[str, Any]) -> None:
        """Save model checkpoint with metadata."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint and return metadata."""
        pass 