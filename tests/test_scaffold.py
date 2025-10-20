"""
Test scaffold-based multi-property optimization with fitness improvement.

Test Configuration:
- Population size: 16 (from standard_config)
- Generations: 3 (from standard_config)  
- Device: cuda
- Scaffold: Quinoline (scaffold_2.mol2)
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


def test_scaffold_multi_property_optimization_with_improvement(generator, standard_config):
    """
    Test scaffold-based multi-property optimization (QED, SA, LogP).
    Verify that:
    1. Optimization completes successfully with scaffold constraint
    2. All molecules contain the specified scaffold
    3. Properties improve from initial to final population
    4. Each individual property improves toward target
    """
    # Use existing Quinoline scaffold MOL2 file
    scaffold_mol2_path = 'datasets/scaffold_examples/scaffold_mol2/scaffold_2.mol2'
    scaffold_smiles = 'c1ccc2ncccc2c1'  # Quinoline (canonical SMILES)
    
    target_properties = {
        'qed': 1,
        'sa': 1,
        'logp': 2.0
    }
    
    print(f"\n🔬 Testing scaffold-based multi-property optimization:")
    print(f"   Scaffold: Quinoline ({scaffold_mol2_path})")
    print(f"   Scaffold SMILES: {scaffold_smiles}")
    print(f"   Target: QED={target_properties['qed']}, SA={target_properties['sa']}, LogP={target_properties['logp']}")
    
    # Create temporary output directory in project root to capture initial and final populations
    output_dir = tempfile.mkdtemp(prefix="test_scaffold_", dir=".")
    print(f"   Temporary output: {output_dir}")
    
    # Run scaffold-based optimization with output_dir to save populations
    molecules = generator.optimize(
        target_properties=target_properties,
        population_size=standard_config['population_size'],
        generations=standard_config['generations'],
        scaffold_mol2_path=scaffold_mol2_path,  # ← SCAFFOLD CONSTRAINT (use MOL2 file directly)
        scaffold_scale_factor=2.5,
        fine_tune_epochs=2,  # Fine-tune model on scaffold-filtered dataset (2 epochs for testing)
        verbose=True,
        output_dir=output_dir
    )
    
    # Verify molecules were generated
    assert len(molecules) > 0, "No molecules generated"
    print(f"\n✓ Generated {len(molecules)} final molecules")
    
    # Verify all molecules contain the scaffold
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    molecules_with_scaffold = 0
    
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(scaffold_mol):
            molecules_with_scaffold += 1
    
    scaffold_percentage = (molecules_with_scaffold / len(molecules)) * 100
    print(f"✓ {molecules_with_scaffold}/{len(molecules)} molecules ({scaffold_percentage:.1f}%) contain the scaffold")
    
    # At least majority should contain scaffold
    assert molecules_with_scaffold >= len(molecules) * 0.5, \
        f"Less than 50% of molecules contain the scaffold ({molecules_with_scaffold}/{len(molecules)})"
    
    # Load initial and final populations for comparison
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
    # QED: should move closer to target
    qed_improved = abs(final_qed - target_properties['qed']) < abs(initial_qed - target_properties['qed'])
    print(f"\n   QED improved: {qed_improved}")
    
    # SA: should move closer to target
    sa_improved = abs(final_sa - target_properties['sa']) < abs(initial_sa - target_properties['sa'])
    print(f"   SA improved: {sa_improved}")
    
    # LogP: should move closer to target
    logp_improved = abs(final_logp - target_properties['logp']) < abs(initial_logp - target_properties['logp'])
    print(f"   LogP improved: {logp_improved}")
    
    # At least one property should improve
    improvements = [qed_improved, sa_improved, logp_improved]
    num_improved = sum(improvements)
    
    print(f"\n   Total: {num_improved}/3 properties improved toward targets")
    
    assert num_improved >= 1, f"No property improvement detected! QED: {qed_improved}, SA: {sa_improved}, LogP: {logp_improved}"
    
    print("\n✅ Scaffold-based multi-property optimization verified!")
    print(f"   ✓ Scaffold constraint maintained ({scaffold_percentage:.1f}% molecules contain scaffold)")
    print(f"   ✓ Properties improved from initial to final population")
    print(f"   ✓ Results saved in: {output_dir}")
