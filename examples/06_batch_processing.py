"""
Example 6: Batch Processing Multiple Targets
=============================================

This example shows how to run multiple optimization jobs in sequence,
useful for systematic exploration or production workflows.

Usage:
    python examples/06_batch_processing.py
"""

from evodiffmol import MoleculeGenerator
from evodiffmol.utils.datasets import General3D
import pandas as pd
from datetime import datetime

def main():
    print("=" * 80)
    print("Example 6: Batch Processing Multiple Optimization Jobs")
    print("=" * 80)
    
    # Load dataset once
    print("\n📦 Loading dataset...")
    dataset = General3D('moses', split='valid', remove_h=True)
    
    # Initialize generator once
    print("🧬 Initializing generator...")
    gen = MoleculeGenerator(
        checkpoint_path="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt",
        model_config="configs/general_without_h.yml",
        ga_config="ga_config/moses_production.yml",
        dataset=dataset,
        verbose=False
    )
    
    # Define batch of optimization jobs
    jobs = [
        {
            'name': 'high_druglikeness',
            'targets': {'qed': 0.95},
            'description': 'Highly drug-like molecules'
        },
        {
            'name': 'lipophilic',
            'targets': {'logp': 4.0, 'qed': 0.8},
            'description': 'Lipophilic compounds'
        },
        {
            'name': 'balanced',
            'targets': {'logp': 2.5, 'qed': 0.9, 'sa': 2.0},
            'description': 'Balanced properties for synthesis'
        },
        {
            'name': 'low_tpsa',
            'targets': {'tpsa': 40, 'qed': 0.85},
            'description': 'Low polar surface area (better permeability)'
        }
    ]
    
    print(f"\n🔄 Running {len(jobs)} optimization jobs...")
    results_summary = []
    
    for i, job in enumerate(jobs, 1):
        print(f"\n{'='*60}")
        print(f"Job {i}/{len(jobs)}: {job['name']}")
        print(f"Description: {job['description']}")
        print(f"Targets: {job['targets']}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Run optimization
        df = gen.optimize(
            target_properties=job['targets'],
            population_size=32,
            generations=3,
            return_dataframe=True,
            output_dir=f"examples/output/06_batch/{job['name']}",
            verbose=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Collect results
        summary = {
            'job_name': job['name'],
            'description': job['description'],
            'num_molecules': len(df),
            'avg_fitness': df['fitness_score'].mean(),
            'duration_sec': duration
        }
        
        # Add property averages
        for prop in job['targets'].keys():
            if prop in df.columns:
                summary[f'avg_{prop}'] = df[prop].mean()
                summary[f'target_{prop}'] = job['targets'][prop]
        
        results_summary.append(summary)
        
        print(f"\n✅ Completed in {duration:.1f} seconds")
        print(f"   Generated: {len(df)} molecules")
        print(f"   Average fitness: {df['fitness_score'].mean():.3f}")
    
    # Create summary report
    print(f"\n{'='*80}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(results_summary)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_file = 'examples/output/06_batch/batch_summary.xlsx'
    summary_df.to_excel(summary_file, index=False)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Calculate total time
    total_time = summary_df['duration_sec'].sum()
    print(f"\n⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n💡 Tips for batch processing:")
    print("   - Reuse generator instance (faster)")
    print("   - Set verbose=False for cleaner output")
    print("   - Use return_dataframe=True for easy analysis")
    print("   - Save results immediately to avoid data loss")


if __name__ == "__main__":
    main()

