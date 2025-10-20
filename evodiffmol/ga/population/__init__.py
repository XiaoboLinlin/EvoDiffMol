"""
Population management for genetic algorithm.
"""

from .population_dataset import GAPopulationDataset, create_data_loader

__all__ = [
    'GAPopulationDataset', 
    'create_data_loader'
] 