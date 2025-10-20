"""
Test API initialization and basic functionality.
"""

import pytest
from evodiffmol import MoleculeGenerator


def test_generator_init(checkpoint_path, model_config_path, ga_config_path, dataset):
    """Test generator initialization."""
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        model_config=model_config_path,
        ga_config=ga_config_path,
        dataset=dataset,
        verbose=False
    )
    
    assert gen is not None
    assert gen.checkpoint_path == checkpoint_path
    assert gen.device == 'cuda'


def test_generator_init_missing_checkpoint():
    """Test that missing checkpoint raises error."""
    with pytest.raises(FileNotFoundError):
        gen = MoleculeGenerator(
            checkpoint_path="nonexistent.pt"
        )


def test_generator_device_selection(checkpoint_path, model_config_path, ga_config_path, dataset):
    """Test device selection."""
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        model_config=model_config_path,
        ga_config=ga_config_path,
        device='cuda',
        dataset=dataset,
        verbose=False
    )
    assert gen.device == 'cuda'


def test_generator_auto_config_detection(checkpoint_path, dataset):
    """Test auto-detection of config files."""
    gen = MoleculeGenerator(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        verbose=False
    )
    
    # Should have auto-detected configs
    assert gen.model_config_path is not None
    assert gen.ga_config_path is not None


