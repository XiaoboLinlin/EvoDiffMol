"""
Example 5: Property Exploration
================================

This example shows how to explore different property targets and compare results.
We'll optimize for different LogP values to understand the trade-offs.

Usage:
    python examples/05_property_exploration.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D
import pandas as pd

def main():
    print("=" * 80)
    print("Example 5: Property Exploration - Different LogP Targets")
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
    
    # Explore different LogP targets
    logp_targets = [1.0, 2.5, 4.0]
    all_results = []
    
    print("\n🔬 Exploring different LogP targets...")
    print("   LogP controls lipophilicity (fat solubility)")
    print("   - Low LogP (~1): More water-soluble")
    print("   - Mid LogP (~2-3): Balanced (typical for drugs)")
    print("   - High LogP (~4+): More fat-soluble")
    
    for logp_target in logp_targets:
        print(f"\n🎯 Optimizing for LogP = {logp_target}...")
        
        df = gen.optimize(
            target_properties={'logp': logp_target, 'qed': 0.9},
            population_size=32,  # Smaller for quick comparison
            generations=3,
            return_dataframe=True,
            output_dir=f'examples/output/05_explore/logp_{logp_target}',
            verbose=False
        )
        
        # Add target column
        df['target_logp'] = logp_target
        all_results.append(df)
        
        print(f"   Generated {len(df)} molecules")
        print(f"   Average LogP: {df['logp'].mean():.2f}")
        print(f"   Average QED: {df['qed'].mean():.2f}")
    
    # Combine and analyze
    print("\n📊 Comparing Results:")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print("\n   Summary by Target:")
    summary = combined_df.groupby('target_logp').agg({
        'logp': ['mean', 'std'],
        'qed': ['mean', 'std'],
        'fitness_score': 'mean'
    }).round(3)
    print(summary)
    
    # Find best molecule for each target
    print("\n🏆 Best molecule for each target:")
    for logp_target in logp_targets:
        subset = combined_df[combined_df['target_logp'] == logp_target]
        best = subset.nlargest(1, 'fitness_score').iloc[0]
        print(f"\n   LogP Target = {logp_target}:")
        print(f"     SMILES: {best['smiles']}")
        print(f"     Actual LogP: {best['logp']:.2f}")
        print(f"     QED: {best['qed']:.2f}")
        print(f"     Fitness: {best['fitness_score']:.3f}")
    
    # Save combined results
    output_file = 'examples/output/05_explore/comparison.xlsx'
    combined_df.to_excel(output_file, index=False)
    print(f"\n💾 All results saved to: {output_file}")
    
    print("\n💡 Try exploring other properties:")
    print("   - Different QED targets (0.6, 0.8, 0.95)")
    print("   - Different SA targets (1.5, 2.5, 3.5)")
    print("   - Different TPSA targets (40, 70, 100)")


if __name__ == "__main__":
    main()

