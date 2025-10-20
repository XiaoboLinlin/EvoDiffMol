"""
Quick smoke test - run this first to verify basic functionality.

This is the most important test! Run it first before full test suite.

Usage:
    pytest tests/quick_test.py -v
"""

import pytest


def test_quick_smoke():
    """Quick smoke test - verify basic import and initialization."""
    # Test imports
    from evodiffmol import MoleculeGenerator
    from evodiffmol.utils.datasets import General3D
    
    print("\n✓ Imports successful")
    
    # Test dataset loading
    dataset = General3D(
        dataset_name='moses',
        split='valid',
        root='datasets',
        remove_h=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} molecules")
    
    # Test generator initialization
    gen = MoleculeGenerator(
        checkpoint_path="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt",
        model_config="configs/general_without_h.yml",
        ga_config="ga_config/moses_production.yml",
        dataset=dataset,
        verbose=False
    )
    
    print("✓ Generator initialized")
    print("✓ Model loaded successfully")
    print("✓ Configurations loaded")
    
    # Validate core components are set up
    assert gen.model is not None, "Model should be loaded"
    assert gen.dataset is not None, "Dataset should be set"
    assert gen.base_ga_config is not None, "GA config should be loaded"
    
    print("\n" + "="*60)
    print("🎉 Quick smoke test PASSED!")
    print("="*60)
    print("\nCore functionality verified:")
    print("  ✅ Package imports working")
    print("  ✅ Dataset loading working (~3-5s validation set)")
    print("  ✅ Generator initialization working")
    print("  ✅ Model and configs loaded")
    print("\nReady for integration!")


if __name__ == "__main__":
    # Allow running directly
    test_quick_smoke()

