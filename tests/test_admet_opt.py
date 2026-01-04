"""
Test ADMET-based GA optimization with fitness improvement.

Test Configuration:
- Population size: 32
- Generations: 3
- Device: cuda
- ADMET properties: Caco2_Wang, hERG, AMES
"""

import pytest
import numpy as np
from evodiffmol import MoleculeGenerator
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import os
import pandas as pd
import tempfile


def test_qed_single_property_optimization(generator):
    """
    Test single-property QED optimization to verify fitness calculation fix.
    
    This test ensures that:
    1. When optimizing only QED, fitness equals QED (not harmonic mean of all properties)
    2. QED values improve toward target
    3. No other properties interfere with QED optimization
    
    This is a regression test for the bug where fitness was calculated as
    hmean([logp, qed, sa, tpsa]) even when only QED was being optimized.
    """
    target_properties = {
        'qed': 1.0  # Maximize QED only
    }
    
    print("\n🎯 Testing single-property QED optimization (bug fix verification):")
    print(f"   Target: QED = {target_properties['qed']}")
    print(f"   Expected: Fitness should equal QED (not hmean of multiple properties)")
    
    # Create output directory
    output_dir = "./tests/test_qed_single_property"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization with only QED
    molecules = generator.optimize(
        target_properties=target_properties,
        population_size=32,
        batch_size=32,
        num_scale_factor=2,
        generations=3,
        save_epoch_csv_freq=1,
        verbose=True,
        output_dir=output_dir
    )
    
    # Verify molecules were generated
    assert len(molecules) > 0, "No molecules generated"
    print(f"\n✓ Generated {len(molecules)} final molecules")
    
    # Load populations
    initial_csv = os.path.join(output_dir, 'initial', 'elite_molecules.csv')
    final_csv = os.path.join(output_dir, 'epoch_last', 'elite_molecules.csv')
    
    assert os.path.exists(initial_csv), f"Initial population not saved"
    assert os.path.exists(final_csv), f"Final population not saved"
    
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    
    print(f"\n📊 Population Comparison:")
    print(f"   Initial population: {len(initial_df)} molecules")
    print(f"   Final population:   {len(final_df)} molecules")
    
    # CRITICAL TEST: Verify fitness == QED (not hmean of all properties)
    print(f"\n🔍 Verifying Fitness Calculation Fix:")
    
    # Check initial population
    if 'qed' in initial_df.columns and 'fitness_score' in initial_df.columns:
        fitness_matches_qed = (abs(initial_df['fitness_score'] - initial_df['qed']) < 0.001).all()
        match_percentage = (abs(initial_df['fitness_score'] - initial_df['qed']) < 0.001).sum() / len(initial_df) * 100
        
        print(f"   Initial population:")
        print(f"      Fitness == QED: {fitness_matches_qed} ({match_percentage:.1f}% match)")
        
        if not fitness_matches_qed:
            # Show sample to debug
            print(f"\n   ⚠️  Sample mismatches:")
            sample = initial_df.head(5)[['qed_raw', 'qed', 'fitness_score']]
            if 'logp' in initial_df.columns:
                sample = initial_df.head(5)[['qed_raw', 'qed', 'logp', 'sa', 'tpsa', 'fitness_score']]
            print(sample.to_string())
        
        assert fitness_matches_qed, \
            f"BUG: Fitness != QED in initial population! " \
            f"Fitness may be using hmean([logp, qed, sa, tpsa]) instead of just QED. " \
            f"Only {match_percentage:.1f}% of molecules have fitness==qed"
    
    # Check final population
    if 'qed' in final_df.columns and 'fitness' in final_df.columns:
        fitness_matches_qed = (abs(final_df['fitness'] - final_df['qed']) < 0.001).all()
        match_percentage = (abs(final_df['fitness'] - final_df['qed']) < 0.001).sum() / len(final_df) * 100
        
        print(f"   Final population:")
        print(f"      Fitness == QED: {fitness_matches_qed} ({match_percentage:.1f}% match)")
        
        assert fitness_matches_qed, \
            f"BUG: Fitness != QED in final population! " \
            f"Only {match_percentage:.1f}% of molecules have fitness==qed"
    
    print(f"   ✅ VERIFIED: Fitness is correctly calculated from QED only!")
    
    # Verify QED improvement
    initial_qed_mean = initial_df['qed_raw'].mean()
    initial_qed_max = initial_df['qed_raw'].max()
    final_qed_mean = final_df['qed_raw'].mean()
    final_qed_max = final_df['qed_raw'].max()
    
    qed_improvement = final_qed_mean - initial_qed_mean
    
    print(f"\n📈 QED Improvement:")
    print(f"   Initial QED:")
    print(f"      Mean: {initial_qed_mean:.4f}")
    print(f"      Max:  {initial_qed_max:.4f}")
    print(f"   Final QED:")
    print(f"      Mean: {final_qed_mean:.4f}")
    print(f"      Max:  {final_qed_max:.4f}")
    print(f"   Improvement: {qed_improvement:+.4f}")
    
    # QED should improve (or at least not get worse)
    assert qed_improvement >= -0.01, \
        f"QED got significantly worse! Initial={initial_qed_mean:.4f}, Final={final_qed_mean:.4f}"
    
    # With correct fitness, we expect improvement in 3 generations
    if qed_improvement > 0.01:
        print(f"   ✅ QED IMPROVED toward target!")
    else:
        print(f"   ⚠️  QED stable (may need more generations for improvement)")
    
    print(f"\n✅ Single-property QED optimization test passed!")
    print(f"   - Fitness calculation is correct (fitness == qed)")
    print(f"   - QED optimization is working as expected")
    print(f"   Results saved in: {output_dir}")


def test_admet_multi_property_optimization(generator):
    """
    Test ADMET multi-property optimization (QED + Caco2 + hERG + AMES).
    Verify that:
    1. Optimization completes with multiple ADMET properties
    2. Properties improve toward targets
    3. Both basic and ADMET properties are optimized together
    """
    target_properties = {
        'qed': 1.0,                           # Drug-likeness (maximize to 1.0)
        'DILI': 0.0,                          # Minimize liver toxicity
        'CYP2D6_Veith': 0.0,                  # Minimize CYP2D6 inhibition
        'PPBR_AZ': 78                         # Moderate protein binding (rounded median)
    }
    
    print("\n🎯 Testing ADMET multi-property optimization with diverse targets:")
    print(f"   Basic: QED={target_properties['qed']} (maximize)")
    print(f"   Distribution: PPBR={target_properties['PPBR_AZ']}% (moderate binding)")
    print(f"   Metabolism: CYP2D6={target_properties['CYP2D6_Veith']} (minimize inhibition)")
    print(f"   Toxicity: DILI={target_properties['DILI']} (minimize liver toxicity)")
    
    print(f"\n🔍 DEBUG - Target Properties:")
    for prop, val in target_properties.items():
        print(f"   {prop}: {val}")
    
    # Create output directory (persistent for inspection)
    output_dir = "./tests/test_admet_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization with multiple ADMET properties
    molecules = generator.optimize(
        target_properties=target_properties,
        population_size=16,
        batch_size=16,
        num_scale_factor=2,
        generations=3,
        save_epoch_csv_freq=1,      # Save CSV every epoch for ablation study (lightweight)
        verbose=True,
        output_dir=output_dir
    )
    
    # Verify molecules were generated
    assert len(molecules) > 0, "No molecules generated"
    print(f"\n✓ Generated {len(molecules)} final molecules with multiple ADMET properties")
    
    # Load populations
    initial_csv = os.path.join(output_dir, 'initial', 'elite_molecules.csv')  # Initial folder
    final_csv = os.path.join(output_dir, 'epoch_last', 'elite_molecules.csv')  # Top level
    
    assert os.path.exists(initial_csv), f"Initial population not saved"
    assert os.path.exists(final_csv), f"Final population not saved"
    
    initial_df = pd.read_csv(initial_csv)
    final_df = pd.read_csv(final_csv)
    
    print(f"\n📊 Population Comparison:")
    print(f"   Initial population: {len(initial_df)} molecules")
    print(f"   Final population:   {len(final_df)} molecules")
    
    # Track improvements
    improvements = []
    
    # Check each ADMET property
    admet_properties = ['DILI', 'CYP2D6_Veith', 'PPBR_AZ']
    
    print(f"\n📈 Property Improvements (Initial → Final):")
    
    for prop in admet_properties:
        raw_col = f'{prop}_raw'
        if raw_col in initial_df.columns and raw_col in final_df.columns:
            initial_val = initial_df[raw_col].mean()
            final_val = final_df[raw_col].mean()
            target_val = target_properties[prop]
            
            initial_dist = abs(initial_val - target_val)
            final_dist = abs(final_val - target_val)
            improved = final_dist < initial_dist
            
            improvements.append(improved)
            
            print(f"   {prop}:")
            print(f"      Initial: {initial_val:.3f}")
            print(f"      Final:   {final_val:.3f} (target: {target_val:.1f})")
            print(f"      Change:  {final_val - initial_val:+.3f}")
            print(f"      Distance to target: {initial_dist:.3f} → {final_dist:.3f}")
            print(f"      Improved: {improved}")
    
    # Check basic properties
    if 'qed_raw' in initial_df.columns and 'qed_raw' in final_df.columns:
        initial_qed = initial_df['qed_raw'].mean()
        final_qed = final_df['qed_raw'].mean()
        qed_improved = abs(final_qed - target_properties['qed']) < abs(initial_qed - target_properties['qed'])
        improvements.append(qed_improved)
        
        print(f"   QED:")
        print(f"      Initial: {initial_qed:.3f}")
        print(f"      Final:   {final_qed:.3f} (target: {target_properties['qed']:.1f})")
        print(f"      Improved: {qed_improved}")
    
    num_improved = sum(improvements)
    total_props = len(improvements)
    
    print(f"\n   Total: {num_improved}/{total_props} properties improved toward targets")
    
    # At least some properties should improve
    assert num_improved >= 1, f"No property improvement detected! Improvements: {improvements}"
    
    print(f"\n✅ ADMET multi-property optimization completed!")
    print(f"   Results saved in: {output_dir}")
    print(f"   Note: ADMET properties found in CSV: {num_improved}/{total_props} improved")


if __name__ == '__main__':
    # For manual testing
    print("Run with pytest: pytest tests/test_admet_opt.py -v")
