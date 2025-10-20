"""
Example 4: Scaffold-Based Optimization
=======================================

This example demonstrates generating molecules that contain a specific
chemical scaffold (substructure). Useful for drug discovery when you
want to maintain a known active scaffold.

Usage:
    python examples/04_scaffold_based.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

def main():
    print("=" * 80)
    print("Example 4: Scaffold-Based Optimization")
    print("=" * 80)
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = General3D('moses', split='valid', remove_h=True)
    
    # Initialize generator
    print("🧬 Initializing generator...")
    gen = MoleculeGenerator(
        checkpoint_path="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt",
        model_config="configs/general_without_h.yml",
        ga_config="ga_config/moses_production.yml",
        dataset=dataset,
        verbose=False
    )
    
    # Define scaffold (benzene ring)
    scaffold_smiles = 'c1ccccc1'  # Benzene
    
    print(f"\n🔬 Scaffold: {scaffold_smiles} (benzene ring)")
    print(f"   All generated molecules will contain this scaffold")
    
    # Optimize with scaffold constraint
    print(f"\n🎯 Optimizing for QED = 0.9 with benzene scaffold...")
    
    molecules = gen.optimize(
        target_properties={'qed': 0.9},
        scaffold_smiles=scaffold_smiles,  # ← Scaffold constraint
        population_size=32,
        generations=3,
        output_dir='examples/output/04_scaffold_benzene',
        verbose=True
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"   Generated {len(molecules)} molecules containing benzene scaffold")
    
    # Show examples
    print(f"\n🧪 Example molecules (all contain {scaffold_smiles}):")
    for i, smiles in enumerate(molecules[:5], 1):
        print(f"   {i}. {smiles}")
    
    print(f"\n📁 Results: examples/output/04_scaffold_benzene/")
    print(f"\n💡 Try different scaffolds:")
    print(f"   - 'c1ccccc1' (benzene)")
    print(f"   - 'C1CCCCC1' (cyclohexane)")  
    print(f"   - 'c1ccc2ccccc2c1' (naphthalene)")
    print(f"   - 'C1=CC=CN=C1' (pyridine)")


if __name__ == "__main__":
    main()

