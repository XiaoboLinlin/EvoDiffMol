"""
Example 1: Single Property Optimization
========================================

This example demonstrates optimizing molecules for a single property (QED).
QED (Quantitative Estimate of Drug-likeness) measures how drug-like a molecule is.

Usage:
    python examples/01_single_property.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

def main():
    print("=" * 80)
    print("Example 1: Single Property Optimization (QED)")
    print("=" * 80)
    
    # Load dataset (validation set for faster loading)
    print("\n📦 Loading dataset...")
    dataset = General3D('moses', split='valid', remove_h=True)
    print(f"   Loaded {len(dataset):,} molecules")
    
    # Initialize generator
    print("\n🧬 Initializing generator...")
    gen = MoleculeGenerator(
        checkpoint_path="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt",
        model_config="configs/general_without_h.yml",
        ga_config="ga_config/moses_production.yml",
        dataset=dataset,
        verbose=False
    )
    print("   Generator initialized")
    
    # Optimize for high QED (drug-likeness)
    print("\n🎯 Optimizing for QED = 0.95...")
    print("   This will generate drug-like molecules")
    
    molecules = gen.optimize(
        target_properties={'qed': 0.95},
        population_size=32,       # Small population for quick demo
        generations=3,            # 3 iterations for fast demo
        output_dir='examples/output/01_qed_optimization',
        verbose=True
    )
    
    # Print results
    print(f"\n✅ Optimization complete!")
    print(f"   Generated {len(molecules)} optimized molecules")
    print(f"\n📁 Results saved to: examples/output/01_qed_optimization")
    print(f"   - Initial population: initial/elite_molecules.csv")
    print(f"   - Final results: epoch_last/elite_molecules.csv")
    print(f"   - 3D structures: epoch_last/mol2_files/")
    
    # Show some example molecules
    print(f"\n🧪 Example molecules:")
    for i, smiles in enumerate(molecules[:5], 1):
        print(f"   {i}. {smiles}")
    
    print("\n💡 Tip: Open the CSV files to see calculated properties!")


if __name__ == "__main__":
    main()

