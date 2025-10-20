"""
Base scoring interface for molecular property evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from rdkit.Chem import rdchem


class ScoringAbstract(ABC):
    """Abstract base class for molecular scoring."""

    @abstractmethod
    def generate_scores(self, mols: List[rdchem.Mol]) -> pd.DataFrame:
        """Generate scores for a list of RDKit molecules.
        
        Args:
            mols: List of RDKit molecule objects
            
        Returns:
            DataFrame with scoring results
        """
        pass

    @abstractmethod
    def get_name_to_function_dict(self) -> Dict[str, callable]:
        """Get dictionary mapping scoring function names to functions.
        
        Returns:
            Dictionary mapping names to scoring functions
        """
        pass

    @property
    @abstractmethod
    def column_names(self) -> List[str]:
        """Get list of column names for scoring results."""
        pass

    @property
    @abstractmethod
    def data_column_name(self) -> str:
        """Get name of the data column."""
        pass

    @property
    @abstractmethod
    def fitness_column_name(self) -> str:
        """Get name of the fitness column."""
        pass 