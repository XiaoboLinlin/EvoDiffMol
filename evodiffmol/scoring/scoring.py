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
    """Molecular scoring class optimized for logP, QED, and synthetic accessibility, with optional ADMET support."""

    def __init__(self, 
                 scoring_names: List[str] = None,
                 selection_names: List[str] = None,
                 scoring_admet_names: Optional[List[str]] = None,
                 property_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 scoring_parameters: Optional[Dict[str, Any]] = None, 
                 data_column_name: str = 'smiles', 
                 fitness_column_name: str = 'fitness', 
                 fitness_function: callable = scipy.stats.hmean):
        """Constructor for MolecularScoring class.

        Args:
            scoring_names: List of scoring function names. Defaults to ['logp', 'qed', 'sa']
            selection_names: List of selection function names. Defaults to same as scoring_names
            scoring_admet_names: Optional list of ADMET property names to predict (requires admet-ai package)
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
        self._scoring_admet_names = scoring_admet_names or []
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
        
        # Setup ADMET model if ADMET properties are requested
        self._admet_model = None
        if self._scoring_admet_names:
            try:
                from admet_ai import ADMETModel
                self._admet_model = ADMETModel()
                print(f"✅ Loaded ADMET-AI model for {len(self._scoring_admet_names)} ADMET properties")
            except ImportError:
                raise ImportError(
                    "ADMET properties require the 'admet-ai' package. "
                    "Install it with: pip install admet-ai"
                )

    def _get_default_property_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default property configuration.
        
        Loads comprehensive ADMET and molecular property configurations from
        property_configs.py, which contains ranges and preferred values derived
        from 2000 MOSES dataset molecules analyzed with ADMET-AI.
        
        Returns:
            Default configuration for all 49 molecular properties
        """
        try:
            from .property_configs import get_all_property_configs
            return get_all_property_configs()
        except ImportError:
            # Fallback to minimal configuration if property_configs not available
            print("Warning: Could not load property_configs.py, using minimal fallback config")
            return {
                'logp': {'range': [-3, 7], 'preferred_value': 2.0},
                'qed': {'range': [0, 1], 'preferred_value': 1.0},
                'sa': {'range': [1, 10], 'preferred_value': 1.0},
                'tpsa': {'range': [0, 200], 'preferred_value': 80.0},
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
    
    def get_name_to_raw_function_dict(self) -> Dict[str, callable]:
        """Get dictionary that maps string to raw (unnormalized) scoring functions.

        Returns:
            Dictionary mapping names to raw scoring functions
        """
        return {
            'logp': lambda mol: self._crippen_mol_logp_with_default(mol, norm=False),
            'qed': self._qed_with_default,
            'sa': lambda mol: self._synthetic_accessibility_with_default(mol, norm=False),
            'synth': lambda mol: self._synthetic_accessibility_with_default(mol, norm=False),  # Alias for sa
            'tpsa': self._tpsa_with_default,
        }

    def generate_scores(self, mols: List[rdkit.Chem.rdchem.Mol]) -> pd.DataFrame:
        """Generate scores for list of RDKit molecules.

        Args:
            mols: List of RDKit molecules

        Returns:
            DataFrame with scores for each molecule (includes both normalized scores, raw values, and ADMET predictions)
        """
        # Initialize results dictionary
        results = {}
        
        # Add SMILES column (vectorized)
        results[self._data_column_name] = [mol_to_canonical_smiles(mol) if mol else None for mol in mols]
        smiles_list = results[self._data_column_name]

        # Get raw function dictionary
        raw_function_dict = self.get_name_to_raw_function_dict()

        # Calculate scores for each scoring function (vectorized with map)
        for scoring_name in self._scoring_names:
            if scoring_name in self._name_to_function:
                scoring_function = self._name_to_function[scoring_name]
                # Vectorized: apply function to all molecules at once
                results[scoring_name] = [scoring_function(mol) if mol else 0.0 for mol in mols]
                
                # Calculate raw values (vectorized)
                if scoring_name in raw_function_dict:
                    raw_function = raw_function_dict[scoring_name]
                    # Default values mapping
                    default_map = {
                        'logp': -3.0, 'qed': 0.0, 
                        'sa': 10.0, 'synth': 10.0, 'tpsa': 200.0
                    }
                    default_val = default_map.get(scoring_name, 0.0)
                    results[f'{scoring_name}_raw'] = [
                        raw_function(mol) if mol else default_val for mol in mols
                    ]

        # Calculate ADMET properties if requested
        if self._scoring_admet_names and self._admet_model is not None:
            try:
                # Get ADMET predictions for all molecules (single batch call - efficient!)
                admet_df = self._admet_model.predict(smiles_list)
                
                # Validate requested properties
                missing_properties = [prop for prop in self._scoring_admet_names if prop not in admet_df.columns]
                if missing_properties:
                    print(f"Warning: Some ADMET properties not found: {missing_properties}")
                    print(f"Available: {list(admet_df.columns)[:10]}...")
                
                # Add ADMET scores (vectorized operations on entire columns)
                for admet_name in self._scoring_admet_names:
                    if admet_name in admet_df.columns:
                        # Store raw values (already a pandas Series - efficient!)
                        raw_values = admet_df[admet_name].tolist()
                        results[f'{admet_name}_raw'] = raw_values
                        
                        # Normalize the values (vectorized if possible)
                        if admet_name in self._property_config:
                            # Use list comprehension instead of loop (more efficient in Python)
                            normalized_values = [
                                self._normalize_score(val, admet_name) 
                                for val in raw_values
                            ]
                            results[admet_name] = normalized_values
                        else:
                            results[admet_name] = admet_df[admet_name].tolist()
                            print(f"Warning: No config for {admet_name}, using raw values")
                            
            except Exception as e:
                print(f"Error predicting ADMET properties: {e}")
                # Add default values efficiently
                num_mols = len(mols)
                for admet_name in self._scoring_admet_names:
                    results[admet_name] = [0.0] * num_mols
                    results[f'{admet_name}_raw'] = [0.0] * num_mols

        # Calculate fitness from selection metrics (vectorized)
        if self._selection_names:
            # Build fitness matrix efficiently
            num_mols = len(mols)
            # Get all selection scores as a 2D structure
            selection_matrix = []
            for selection_name in self._selection_names:
                if selection_name in results:
                    selection_matrix.append(results[selection_name])
            
            if selection_matrix:
                # Transpose to get scores per molecule
                # Calculate fitness for all molecules at once
                fitness_scores = []
                for i in range(num_mols):
                    mol_scores = [matrix[i] for matrix in selection_matrix]
                    try:
                        fitness_scores.append(self._fitness_function(mol_scores))
                    except:
                        fitness_scores.append(0.0)
                results[self._fitness_column_name] = fitness_scores
            else:
                results[self._fitness_column_name] = [0.0] * num_mols

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
        # Extract SMILES strings (vectorized)
        smiles_list = [data.smiles for data in population if data is not None and hasattr(data, 'smiles') and data.smiles]
        
        # Generate scores directly from SMILES (single batch call)
        scores_df = self.generate_scores_from_smiles(smiles_list)
        
        # Get all column names once
        all_columns = scores_df.columns.tolist()
        scoring_columns = [col for col in all_columns if col in self._scoring_names]
        raw_columns = [f'{name}_raw' for name in self._scoring_names if f'{name}_raw' in all_columns]
        
        # Update Data objects with scores (vectorized access to DataFrame)
        for i, data in enumerate(population):
            if data is None or not (hasattr(data, 'smiles') and data.smiles) or i >= len(scores_df):
                if data is not None:
                    data.fitness_score = torch.tensor([0.0], dtype=torch.float)
                continue
            
            row = scores_df.iloc[i]
            
            # Set fitness score
            data.fitness_score = torch.tensor([row[self.fitness_column_name]], dtype=torch.float)
            
            # Store normalized property scores (batch operation)
            for col_name in scoring_columns:
                setattr(data, col_name, torch.tensor([row[col_name]], dtype=torch.float))
            
            # Store raw property scores (batch operation)
            for col_name in raw_columns:
                setattr(data, col_name, torch.tensor([row[col_name]], dtype=torch.float))
            
            # Store ADMET scores if present
            if self._scoring_admet_names:
                for admet_name in self._scoring_admet_names:
                    if admet_name in all_columns:
                        setattr(data, admet_name, torch.tensor([row[admet_name]], dtype=torch.float))
                    raw_col = f'{admet_name}_raw'
                    if raw_col in all_columns:
                        setattr(data, raw_col, torch.tensor([row[raw_col]], dtype=torch.float))

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
        
        All properties use preferred_value as the target. The normalization 
        calculates distance from the target and converts to a score in [0, 1].
        
        Args:
            value: Raw property value
            property_name: Name of the property
            
        Returns:
            Normalized score in [0, 1] range (1 = at target, 0 = furthest from target)
        """
        if property_name not in self._property_config:
            return value
            
        config = self._property_config[property_name]
        prop_range = config.get('range', [0, 1])
        preferred_value = config.get('preferred_value')
        
        # Clamp value to range
        value = max(prop_range[0], min(value, prop_range[1]))
        
        # Calculate normalized score based on distance from preferred_value
        if preferred_value is not None:
            # Calculate maximum possible distance from target
            max_dist = max(preferred_value - prop_range[0], prop_range[1] - preferred_value)
            # Calculate actual distance
            dist_from_target = abs(value - preferred_value)
            # Normalize: closer to target = higher score
            if max_dist > 0:
                return max(0.0, 1.0 - (dist_from_target / max_dist))
            else:
                return 1.0 if value == preferred_value else 0.0
        else:
            # Fallback: no preferred value, just normalize to range
            range_span = prop_range[1] - prop_range[0]
            if range_span > 0:
                return (value - prop_range[0]) / range_span
            else:
                return 1.0

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

 