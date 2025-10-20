"""
Example 3: DataFrame Output
============================

This example shows how to get results as a pandas DataFrame with all
calculated properties, making it easy to analyze and filter results.

Usage:
    python examples/03_dataframe_output.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D
import pandas as pd

def main():
    print("=" * 80)
    print("Example 3: DataFrame Output with Property Analysis")
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
    
    # Optimize and return DataFrame
    print("\n🎯 Optimizing for LogP = 3.0 and QED = 0.9...")
    
    df = gen.optimize(
        target_properties={'logp': 3.0, 'qed': 0.9},
        population_size=32,
        generations=3,
        return_dataframe=True,  # ← Return DataFrame instead of list
        output_dir='examples/output/03_dataframe'
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"   Generated {len(df)} molecules")
    
    # Analyze results
    print("\n📊 Results Summary:")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Property Statistics:")
    print(df[['logp', 'qed', 'fitness_score']].describe())
    
    # Find best molecules
    print(f"\n🏆 Top 5 molecules by fitness:")
    top5 = df.nlargest(5, 'fitness_score')
    print(top5[['smiles', 'fitness_score', 'logp', 'qed']])
    
    # Filter by criteria
    print(f"\n🔍 Molecules with LogP > 2.5 and QED > 0.85:")
    filtered = df[(df['logp'] > 2.5) & (df['qed'] > 0.85)]
    print(f"   Found {len(filtered)} molecules")
    if len(filtered) > 0:
        print(filtered.head()[['smiles', 'logp', 'qed']])
    
    # Save to Excel
    output_file = 'examples/output/03_dataframe/results_analysis.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    print("\n💡 You can now use pandas to analyze, filter, and visualize results!")


if __name__ == "__main__":
    main()

