"""
Example 2: Multi-Property Optimization
=======================================

This example demonstrates optimizing for multiple properties simultaneously:
- LogP (lipophilicity): Target = 2.5
- QED (drug-likeness): Target = 0.9
- SA (synthetic accessibility): Target = 2.0 (lower is easier to synthesize)

Usage:
    python examples/02_multi_property.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D

def main():
    print("=" * 80)
    print("Example 2: Multi-Property Optimization")
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
    
    # Define target properties
    target_properties = {
        'logp': 2.5,   # Good permeability (typical for drugs: 1-3)
        'qed': 0.9,    # High drug-likeness
        'sa': 2.0      # Easy to synthesize (1=very easy, 10=very hard)
    }
    
    print(f"\n🎯 Optimizing for multiple properties:")
    print(f"   • LogP  = {target_properties['logp']:.1f} (lipophilicity)")
    print(f"   • QED   = {target_properties['qed']:.1f} (drug-likeness)")
    print(f"   • SA    = {target_properties['sa']:.1f} (synthetic accessibility)")
    
    # Run optimization
    molecules = gen.optimize(
        target_properties=target_properties,
        population_size=32,
        generations=3,
        output_dir='examples/output/02_multi_property',
        verbose=True
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"   Generated {len(molecules)} optimized molecules")
    print(f"\n📁 Results: examples/output/02_multi_property/epoch_last/elite_molecules.csv")
    
    # Show examples
    print(f"\n🧪 Example molecules:")
    for i, smiles in enumerate(molecules[:5], 1):
        print(f"   {i}. {smiles}")
    
    print("\n💡 Check the CSV file to see how close each molecule is to targets!")


if __name__ == "__main__":
    main()

