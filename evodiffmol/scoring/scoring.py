"""
Molecular scoring classes for genetic algorithm optimization.
Combines base scoring functionality with specialized logP, QED, and SA scoring.
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
import numpy as np
import math
import gzip
import pickle
import re
import scipy.stats
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from .abstract_scoring import ScoringAbstract
import torch


def smiles_to_mol(smiles: str) -> Optional[rdkit.Chem.rdchem.Mol]:
    """Convert SMILES to RDKit molecule.

    Args:
        smiles: SMILES string representation of a molecule

    Returns:
        RDKit molecule object or None if conversion fails
    """
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True)
    except:
        return None


def mol_to_canonical_smiles(mol: rdkit.Chem.rdchem.Mol) -> Optional[str]:
    """Convert RDKit molecule to canonical SMILES string.

    Args:
        mol: RDKit molecule object

    Returns:
        Canonical SMILES string or None if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def remap(x: float, x_min: float, x_max: float) -> float:
    """Translate and scale a given input to [0, 1] range.

    Args:
        x: Original value
        x_min: Value to subtract and lower bound for scale
        x_max: Upper bound for scale

    Returns:
        Translated and scaled value
    """
    return (x - x_min) / (x_max - x_min)


class MolecularScoring(ScoringAbstract):
    """Molecular scoring class optimized for logP, QED, and synthetic accessibility."""

    def __init__(self, 
                 scoring_names: List[str] = None,
                 selection_names: List[str] = None,
                 property_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 scoring_parameters: Optional[Dict[str, Any]] = None, 
                 data_column_name: str = 'smiles', 
                 fitness_column_name: str = 'fitness', 
                 fitness_function: callable = scipy.stats.hmean):
        """Constructor for MolecularScoring class.

        Args:
            scoring_names: List of scoring function names. Defaults to ['logp', 'qed', 'sa']
            selection_names: List of selection function names. Defaults to same as scoring_names
            property_config: Configuration for property ranges and preferences
            scoring_parameters: Parameters for scoring functions (e.g., SA model path)
            data_column_name: Name for data column
            fitness_column_name: Name for fitness column
            fitness_function: Function to aggregate selection scores into fitness
        """
        super().__init__()

        # Set defaults if not provided
        if scoring_names is None:
            scoring_names = ['logp', 'qed', 'sa']
        if selection_names is None:
            selection_names = scoring_names.copy()

        # Setup scoring parameters
        if scoring_parameters is None:
            scoring_parameters = {}

        # Store variables
        self._scoring_names = scoring_names
        self._selection_names = selection_names
        self._data_column_name = data_column_name
        self._fitness_column_name = fitness_column_name
        self._fitness_function = fitness_function
        
        # Store property configuration
        self._property_config = property_config or self._get_default_property_config()

        # Dictionary storing mapping from names to scoring functions
        self._name_to_function = self.get_name_to_function_dict()

        # Check that data and fitness column names are not part of possible scoring functions
        if self.data_column_name in self._name_to_function:
            raise ValueError(f'Error: data column name {self.data_column_name} cannot be in {list(self._name_to_function.keys())}')

        if self.fitness_column_name in self._name_to_function:
            raise ValueError(f'Error: fitness column name {self.fitness_column_name} cannot be in {list(self._name_to_function.keys())}')

        # Setup for synthetic accessibility - use RDKit's built-in SA scorer
        if 'sa' in scoring_names or 'synth' in scoring_names:
            from rdkit.Contrib.SA_Score import sascorer
            self._sa_scorer = sascorer
            print("✅ Using RDKit SA_Score module for synthetic accessibility scoring")

    def _get_default_property_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default property configuration.
        
        Supports two preference modes for optimization:
        - preferred_range: [min, max] - optimal values within this range
        - preferred_value: target_value - optimal single target value
        
        Returns:
            Default configuration for molecular properties
        """
        return {
            'logp': {
                'range': [-4, 7], 
                'preferred_range': [1.0, 3.0],  # Use preferred_range OR preferred_value, not both
                # 'preferred_value': 2.0,  # Alternative: target a specific logP value
                'higher_is_better': None  # Preference-based
            },
            'qed': {
                'range': [0, 1], 
                'higher_is_better': True
            },
            'sa': {
                'range': [1, 10], 
                'higher_is_better': False  # Lower SA scores are better
            },
            'tpsa': {
                'range': [0, 188],
                'preferred_value': 90,  # Balanced drug-like value
                'higher_is_better': False  # Lower TPSA is generally better for permeability
            }
        }

    def get_name_to_function_dict(self) -> Dict[str, callable]:
        """Get dictionary that maps string to scoring functions.

        Returns:
            Dictionary mapping names to scoring functions
        """
        return {
            'logp': self._compute_logp_normalized,
            'qed': self._compute_qed_normalized,
            'sa': self._compute_sa_normalized,
            'synth': self._compute_sa_normalized,  # Alias for sa
            'tpsa': self._compute_tpsa_normalized,
        }

    def generate_scores(self, mols: List[rdkit.Chem.rdchem.Mol]) -> pd.DataFrame:
        """Generate scores for list of RDKit molecules.

        Args:
            mols: List of RDKit molecules

        Returns:
            DataFrame with scores for each molecule
        """
        # Initialize results dictionary
        results = {}
        
        # Add SMILES column
        smiles_list = []
        for mol in mols:
            if mol is not None:
                smiles_list.append(mol_to_canonical_smiles(mol))
            else:
                smiles_list.append(None)
        results[self._data_column_name] = smiles_list

        # Calculate scores for each scoring function
        for scoring_name in self._scoring_names:
            if scoring_name in self._name_to_function:
                scoring_function = self._name_to_function[scoring_name]
                scores = []
                for mol in mols:
                    if mol is not None:
                        scores.append(scoring_function(mol))
                    else:
                        scores.append(0.0)  # Default score for invalid molecules
                results[scoring_name] = scores

        # Calculate fitness from selection metrics
        if self._selection_names:
            fitness_scores = []
            for i in range(len(mols)):
                selection_scores = []
                for selection_name in self._selection_names:
                    if selection_name in results:
                        selection_scores.append(results[selection_name][i])
                
                if selection_scores:
                    try:
                        fitness = self._fitness_function(selection_scores)
                    except:
                        fitness = 0.0
                else:
                    fitness = 0.0
                fitness_scores.append(fitness)
            
            results[self._fitness_column_name] = fitness_scores

        return pd.DataFrame(results)

    def generate_scores_from_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """Generate scores for list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with scores for each molecule
        """
        # Convert SMILES to RDKit molecules
        mols = [smiles_to_mol(smiles) for smiles in smiles_list]
        return self.generate_scores(mols)

    def evaluate_population(self, population: List['Data']) -> None:
        """Evaluate fitness scores for a population of Data objects.

        Args:
            population: List of Data objects to evaluate
        """
        # Extract SMILES strings from Data objects
        smiles_list = [data.smiles for data in population if data is not None and hasattr(data, 'smiles') and data.smiles]
        
        # Generate scores directly from SMILES
        scores_df = self.generate_scores_from_smiles(smiles_list)
        
        # Update Data object fitness scores
        for i, data in enumerate(population):
            if data is None:
                continue  # Skip None values
            if hasattr(data, 'smiles') and data.smiles and i < len(scores_df):
                data.fitness_score = torch.tensor([scores_df.iloc[i][self.fitness_column_name]], dtype=torch.float)
                
                # Store individual property scores
                for scoring_name in self._scoring_names:
                    if scoring_name in scores_df.columns:
                        setattr(data, scoring_name, torch.tensor([scores_df.iloc[i][scoring_name]], dtype=torch.float))
            else:
                data.fitness_score = torch.tensor([0.0], dtype=torch.float)

    @property
    def column_names(self) -> List[str]:
        """Get list of column names."""
        columns = [self._data_column_name] + self._scoring_names
        if self._selection_names:
            columns.append(self._fitness_column_name)
        return columns

    @property
    def selection_names(self) -> List[str]:
        """Get list of selection names."""
        return self._selection_names

    @property
    def data_column_name(self) -> str:
        """Get data column name."""
        return self._data_column_name

    @property
    def fitness_column_name(self) -> str:
        """Get fitness column name."""
        return self._fitness_column_name

    def _normalize_score(self, value: float, property_name: str) -> float:
        """Normalize a score based on property configuration.
        
        Args:
            value: Raw property value
            property_name: Name of the property
            
        Returns:
            Normalized score in [0, 1] range
        """
        if property_name not in self._property_config:
            return value
            
        config = self._property_config[property_name]
        prop_range = config.get('range', [0, 1])
        preferred_range = config.get('preferred_range')
        preferred_value = config.get('preferred_value')
        higher_is_better = config.get('higher_is_better', True)
        
        # Clamp to range
        value = max(prop_range[0], min(value, prop_range[1]))
        
        if preferred_value is not None:
            # Preference-based scoring with a single target value
            max_dist = max(preferred_value - prop_range[0], prop_range[1] - preferred_value)
            dist_from_target = abs(value - preferred_value)
            if max_dist > 0:
                return max(0.0, 1.0 - (dist_from_target / max_dist))
            else:
                return 1.0 if value == preferred_value else 0.0
        elif preferred_range:
            # Preference-based scoring with a range (e.g., for logP)
            if preferred_range[0] <= value <= preferred_range[1]:
                # Value is in preferred range - score based on distance from center
                center = (preferred_range[0] + preferred_range[1]) / 2
                max_dist = (preferred_range[1] - preferred_range[0]) / 2
                dist_from_center = abs(value - center)
                return 1.0 - (dist_from_center / max_dist)
            else:
                # Value is outside preferred range - penalize based on distance
                if value < preferred_range[0]:
                    dist = preferred_range[0] - value
                    max_dist = preferred_range[0] - prop_range[0]
                else:
                    dist = value - preferred_range[1]
                    max_dist = prop_range[1] - preferred_range[1]
                
                if max_dist > 0:
                    return max(0.0, 1.0 - (dist / max_dist))
                else:
                    return 0.0
        else:
            # Simple range-based scoring
            normalized = (value - prop_range[0]) / (prop_range[1] - prop_range[0])
            if not higher_is_better:
                normalized = 1.0 - normalized
            return max(0.0, min(1.0, normalized))

    def _compute_logp_normalized(self, mol) -> float:
        """Compute normalized LogP score.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Normalized LogP score
        """
        raw_logp = self._crippen_mol_logp_with_default(mol, norm=False)
        return self._normalize_score(raw_logp, 'logp')
    
    def _compute_qed_normalized(self, mol) -> float:
        """Compute normalized QED score.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Normalized QED score
        """
        raw_qed = self._qed_with_default(mol)
        return self._normalize_score(raw_qed, 'qed')
    
    def _compute_sa_normalized(self, mol) -> float:
        """Compute normalized synthetic accessibility score.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Normalized SA score (higher is better)
        """
        raw_sa = self._synthetic_accessibility_with_default(mol, norm=False)
        return self._normalize_score(raw_sa, 'sa')
    
    def _compute_tpsa_normalized(self, mol) -> float:
        """Compute normalized TPSA (Topological Polar Surface Area) score.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Normalized TPSA score
        """
        raw_tpsa = self._tpsa_with_default(mol)
        return self._normalize_score(raw_tpsa, 'tpsa')

    def _qed_with_default(self, mol: rdkit.Chem.rdchem.Mol, default: float = 0.0) -> float:
        """Calculate QED (drug-likeness) with default value for invalid molecules.

        Args:
            mol: RDKit molecule
            default: Default value to return if calculation fails

        Returns:
            QED score
        """
        if mol is None:
            return default
        try:
            return QED.qed(mol)
        except:
            return default

    def _crippen_mol_logp_with_default(self, mol: rdkit.Chem.rdchem.Mol, default: float = -3.0, norm: bool = True) -> float:
        """Calculate LogP with default value for invalid molecules.

        Args:
            mol: RDKit molecule
            default: Default value to return if calculation fails
            norm: Whether to normalize the score

        Returns:
            LogP score
        """
        if mol is None:
            return default
        try:
            logp = Crippen.MolLogP(mol)
            if norm:
                # Normalize LogP to [0, 1] range assuming typical range [-4, 7]
                return remap(logp, -4, 7)
            return logp
        except:
            return default

    def _synthetic_accessibility_with_default(self, mol: rdkit.Chem.rdchem.Mol, default: float = 10, norm: bool = True) -> float:
        """Calculate synthetic accessibility with default value for invalid molecules.

        Args:
            mol: RDKit molecule
            default: Default value to return if calculation fails
            norm: Whether to normalize the score (lower SA is better, so we invert)

        Returns:
            Synthetic accessibility score
        """
        if mol is None:
            return 0.0 if norm else default
            
        sa_score = self._sa_scorer.calculateScore(mol)
        
        if norm:
            # SA scores typically range from 1-10, where 1 is easy to synthesize
            # We invert and normalize so higher scores are better
            normalized = (10 - sa_score) / 9  # Invert and scale to [0, 1]
            return max(0.0, normalized)
        return sa_score
    
    def _tpsa_with_default(self, mol: rdkit.Chem.rdchem.Mol, default: float = 200.0) -> float:
        """Calculate TPSA (Topological Polar Surface Area) with default value for invalid molecules.

        Args:
            mol: RDKit molecule
            default: Default value to return if calculation fails (200.0 = poor drug-likeness)

        Returns:
            TPSA value in Ų (Angstrom squared)
        """
        if mol is None:
            return default
        try:
            return Descriptors.TPSA(mol)
        except:
            return default

 