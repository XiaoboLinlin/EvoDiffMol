"""
Genetic Algorithm module for molecular generation using diffusion models.

This module provides a modular architecture for implementing genetic algorithms
with adaptive training capabilities for molecular diffusion models.
"""

from .core.base import GAConfig
from .core.config_loader import GAConfigLoader
from .population.population_dataset import GAPopulationDataset, create_data_loader
from .core.trainer import GeneticTrainer

__version__ = "1.0.0"
__all__ = [
    'GAConfig', 
    'GAConfigLoader',
    'GAPopulationDataset',
    'create_data_loader',
    'GeneticTrainer'
] 