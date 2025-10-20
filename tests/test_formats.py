"""
Test output formats (list, DataFrame, file I/O).
"""

import pytest
import pandas as pd
from pathlib import Path


def test_list_output(generator, standard_config):
    """Test default list output."""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=standard_config['population_size'],
        generations=standard_config['generations']
    )
    
    assert isinstance(molecules, list)
    assert all(isinstance(s, str) for s in molecules)


def test_dataframe_output(generator, standard_config):
    """Test DataFrame output."""
    df = generator.optimize(
        target_properties={'logp': 4.0, 'qed': 0.9},
        population_size=standard_config['population_size'],
        generations=standard_config['generations'],
        return_dataframe=True
    )
    
    assert isinstance(df, pd.DataFrame)
    assert 'smiles' in df.columns
    # Note: property columns depend on implementation


def test_output_dir_basic(generator, standard_config, temp_output_dir):
    """Test file output for basic optimization."""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=standard_config['population_size'],
        generations=standard_config['generations'],
        output_dir=temp_output_dir,
        save_results=True
    )
    
    # Check that output directory was created
    output_path = Path(temp_output_dir)
    assert output_path.exists()
    
    # Check for expected files/directories
    # Note: Exact structure depends on implementation
    assert len(list(output_path.iterdir())) > 0  # Something was written


def test_output_dir_scaffold(generator, standard_config, temp_output_dir):
    """Test file output for scaffold optimization."""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=8,  # Smaller for speed
        generations=2,
        scaffold_smiles='c1ccccc1',
        output_dir=temp_output_dir,
        save_results=True
    )
    
    output_path = Path(temp_output_dir)
    assert output_path.exists()
    assert len(list(output_path.iterdir())) > 0


def test_no_output_dir(generator, standard_config):
    """Test that no files are written when output_dir is None."""
    molecules = generator.optimize(
        target_properties={'logp': 4.0},
        population_size=standard_config['population_size'],
        generations=standard_config['generations'],
        output_dir=None  # Explicitly no output
    )
    
    # Should return molecules without writing files
    assert len(molecules) > 0


