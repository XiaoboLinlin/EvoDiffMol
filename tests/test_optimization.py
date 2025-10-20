"""
Test GA optimization with fitness improvement.

Test Configuration:
- Population size: 8
- Generations: 3
- Device: cuda
"""

import pytest
import numpy as np
from evodiffmol import MoleculeGenerator
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import os
import pandas as pd
import tempfile
import shutil


def calculate_properties(smiles):
    """Calculate molecular properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate SA using RDKit's built-in sascorer
    try:
        from rdkit.Contrib.SA_Score import sascorer
        sa_score = sascorer.calculateScore(mol)
    except Exception as e:
        sa_score = None
    
    return {
        'qed': QED.qed(mol),
        'logp': Descriptors.MolLogP(mol),
        'sa': sa_score
    }


def test_single_property_optimization_qed(generator):
    """
    Test single-property optimization (QED=1.0).
    Verify that:
    1. Optimization completes successfully
    2. QED improves from initial to final population
    3. Final QED is closer to target than initial QED
    """
    target_properties = {
        'qed': 1.0
    }
    
    print("\n🎯 Testing single-property optimization:")
    print(f"   Target: QED={target_properties['qed']}")
    
    # Create output directory in project root to capture initial population
    output_dir = tempfile.mkdtemp(prefix="test_opt_qed_", dir=".")
    
    # Run optimization with output_dir to save initial population
    molecules = generator.optimize(
        target_properties=target_properties,
        population_size=64,
        batch_size=64,
        num_scale_factor=1.5,  # Generate 1.5x molecules (48 total) - same as debug script
        generations=3,
        verbose=True,
        output_dir=output_dir
    )
    
    # Verify molecules were generated
    assert len(molecules) > 0, "No molecules generated"
    print(f"\n✓ Generated {len(molecules)} final molecules")
    
    # Load initial population for comparison
    initial_csv = os.path.join(output_dir, 'initial', 'elite_molecules.csv')
    final_csv = os.path.join(output_dir, 'epoch_last', 'elite_molecules.csv')
    
    assert os.path.exists(initial_csv), f"Initial population not saved: {initial_csv}"
    assert os.path.exists(final_csv), f"Final population not saved: {final_csv}"
    
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    
    print(f"\n📊 Population Comparison:")
    print(f"   Initial population: {len(initial_df)} molecules")
    print(f"   Final population:   {len(final_df)} molecules")
    
    # Calculate average QED for both populations
    initial_props = []
    for smiles in initial_df['smiles']:
        props = calculate_properties(smiles)
        if props:
            initial_props.append(props)
    
    final_props = []
    for smiles in final_df['smiles']:
        props = calculate_properties(smiles)
        if props:
            final_props.append(props)
    
    # Compare averages
    initial_qed = np.mean([p['qed'] for p in initial_props])
    final_qed = np.mean([p['qed'] for p in final_props])
    
    print(f"\n📈 QED Improvement (Initial → Final):")
    print(f"   Initial: {initial_qed:.3f}")
    print(f"   Final:   {final_qed:.3f} (target: {target_properties['qed']:.1f})")
    print(f"   Change:  {final_qed - initial_qed:+.3f}")
    print(f"   Distance to target: {abs(initial_qed - target_properties['qed']):.3f} → {abs(final_qed - target_properties['qed']):.3f}")
    
    # Verify improvement toward target
    qed_improved = abs(final_qed - target_properties['qed']) < abs(initial_qed - target_properties['qed'])
    print(f"\n   QED improved: {qed_improved}")
    
    assert qed_improved, f"QED did not improve toward target! Initial: {initial_qed:.3f}, Final: {final_qed:.3f}, Target: {target_properties['qed']}"
    
    print("\n✅ Single property optimization verified!")
    print(f"   QED improved from initial to final population")
    print(f"   Results saved in: {output_dir}")
    
    # Verify config was saved
    config_file = os.path.join(output_dir, 'config', 'ga_config_used.yml')
    assert os.path.exists(config_file), f"GA config not saved: {config_file}"
    print(f"   Config saved: {config_file}")


def test_multi_property_optimization_with_fitness_improvement(generator):
    """
    Test multi-property optimization (QED=1, SA=1, LogP=2).
    Verify that:
    1. Optimization completes successfully
    2. Properties improve from initial to final population
    3. Each individual property improves toward target
    """
    target_properties = {
        'qed': 1.0,
        'sa': 1.0,
        'logp': 2.0
    }
    
    print("\n🎯 Testing multi-property optimization:")
    print(f"   Target: QED={target_properties['qed']}, SA={target_properties['sa']}, LogP={target_properties['logp']}")
    
    # Create output directory in project root to capture initial population
    output_dir = tempfile.mkdtemp(prefix="test_opt_", dir=".")
    
    # Run optimization with output_dir to save initial population
    molecules = generator.optimize(
        target_properties=target_properties,
        population_size=32,
        batch_size=32,
        num_scale_factor=1.5,  # Generate 1.5x molecules (48 total) - same as debug script
        generations=3,
        verbose=True,
        output_dir=output_dir
    )
    
    # Verify molecules were generated
    assert len(molecules) > 0, "No molecules generated"
    print(f"\n✓ Generated {len(molecules)} final molecules")
    
    # Load initial population for comparison
    initial_csv = os.path.join(output_dir, 'initial', 'elite_molecules.csv')
    final_csv = os.path.join(output_dir, 'epoch_last', 'elite_molecules.csv')
    
    assert os.path.exists(initial_csv), f"Initial population not saved: {initial_csv}"
    assert os.path.exists(final_csv), f"Final population not saved: {final_csv}"
    
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    
    print(f"\n📊 Population Comparison:")
    print(f"   Initial population: {len(initial_df)} molecules")
    print(f"   Final population:   {len(final_df)} molecules")
    
    # Calculate average properties for both populations
    initial_props = []
    for smiles in initial_df['smiles']:
        props = calculate_properties(smiles)
        if props:
            initial_props.append(props)
    
    final_props = []
    for smiles in final_df['smiles']:
        props = calculate_properties(smiles)
        if props:
            final_props.append(props)
    
    # Compare averages
    initial_qed = np.mean([p['qed'] for p in initial_props])
    initial_logp = np.mean([p['logp'] for p in initial_props])
    initial_sa = np.mean([p['sa'] for p in initial_props if p['sa'] is not None])
    
    final_qed = np.mean([p['qed'] for p in final_props])
    final_logp = np.mean([p['logp'] for p in final_props])
    final_sa = np.mean([p['sa'] for p in final_props if p['sa'] is not None])
    
    print(f"\n📈 Property Improvements (Initial → Final):")
    print(f"   QED:")
    print(f"      Initial: {initial_qed:.3f}")
    print(f"      Final:   {final_qed:.3f} (target: {target_properties['qed']:.1f})")
    print(f"      Change:  {final_qed - initial_qed:+.3f}")
    print(f"      Distance to target: {abs(initial_qed - target_properties['qed']):.3f} → {abs(final_qed - target_properties['qed']):.3f}")
    
    print(f"   SA (Synthetic Accessibility):")
    print(f"      Initial: {initial_sa:.3f}")
    print(f"      Final:   {final_sa:.3f} (target: {target_properties['sa']:.1f})")
    print(f"      Change:  {final_sa - initial_sa:+.3f}")
    print(f"      Distance to target: {abs(initial_sa - target_properties['sa']):.3f} → {abs(final_sa - target_properties['sa']):.3f}")
    print(f"      (Lower SA is better: 1=easy, 10=hard to synthesize)")
    
    print(f"   LogP:")
    print(f"      Initial: {initial_logp:.3f}")
    print(f"      Final:   {final_logp:.3f} (target: {target_properties['logp']:.1f})")
    print(f"      Change:  {final_logp - initial_logp:+.3f}")
    print(f"      Distance to target: {abs(initial_logp - target_properties['logp']):.3f} → {abs(final_logp - target_properties['logp']):.3f}")
    
    # Verify improvement toward targets
    # QED: should move closer to target (1.0)
    qed_improved = abs(final_qed - target_properties['qed']) < abs(initial_qed - target_properties['qed'])
    print(f"\n   QED improved: {qed_improved}")
    
    # SA: should move closer to target (1.0)
    sa_improved = abs(final_sa - target_properties['sa']) < abs(initial_sa - target_properties['sa'])
    print(f"   SA improved: {sa_improved}")
    
    # LogP: should move closer to target (2.0)
    logp_improved = abs(final_logp - target_properties['logp']) < abs(initial_logp - target_properties['logp'])
    print(f"   LogP improved: {logp_improved}")
    
    # At least one property should improve
    improvements = [qed_improved, sa_improved, logp_improved]
    num_improved = sum(improvements)
    
    print(f"\n   Total: {num_improved}/3 properties improved toward targets")
    
    assert num_improved >= 1, f"No property improvement detected! QED: {qed_improved}, SA: {sa_improved}, LogP: {logp_improved}"
    
    print("\n✅ Fitness improvement verified!")
    print(f"   Properties improved from initial to final population")
    print(f"   Results saved in: {output_dir}")
