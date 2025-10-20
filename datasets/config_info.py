#!/usr/bin/env python3
"""
Config Info Generator for GuacaMol and MOSES Datasets

This script analyzes ALL splits (train, valid, test) of the processed NPZ datasets 
and generates configuration dictionaries compatible with configs/datasets_config.py format.

Usage:
    python config_info.py

Output:
    Prints formatted Python dictionaries for:
    - guacamol_with_h (combined stats from all splits)
    - guacamol_without_h (combined stats from all splits)
    - moses_with_h (combined stats from all splits)
    - moses_without_h (combined stats from all splits)

The output can be copied directly into configs/datasets_config.py
"""

import os
import sys
import numpy as np
from collections import Counter, defaultdict

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def analyze_npz_dataset_splits(base_path, dataset_name):
    """Analyze all splits (train, valid, test) of a dataset and combine statistics."""
    splits = ['train.npz', 'valid.npz', 'test.npz']
    
    print(f"📊 Analyzing {dataset_name}: {base_path}")
    
    all_atoms = []
    n_nodes_count = Counter()
    total_molecules = 0
    found_splits = []
    
    for split in splits:
        npz_path = os.path.join(base_path, split)
        if not os.path.exists(npz_path):
            print(f"   ⚠️  {split} not found, skipping...")
            continue
            
        try:
            data = np.load(npz_path)
            charges = data['charges']  # atomic numbers
            num_atoms = data['num_atoms']
            
            split_molecules = len(charges)
            total_molecules += split_molecules
            found_splits.append(f"{split}: {split_molecules}")
            
            for i, n_atoms in enumerate(num_atoms):
                mol_charges = charges[i][:n_atoms]
                all_atoms.extend(mol_charges)
                n_nodes_count[n_atoms] += 1
                
        except Exception as e:
            print(f"   ❌ Error loading {split}: {e}")
            continue
    
    if not all_atoms:
        print(f"   ❌ No valid data found for {dataset_name}")
        return None
    
    print(f"   📦 Loaded {total_molecules} total molecules from: {', '.join(found_splits)}")
    
    # Get unique atomic numbers
    unique_atoms = sorted(set(all_atoms))
    atom_counts = Counter(all_atoms)
    
    print(f"   🧪 Found atoms: {unique_atoms}")
    print(f"   📏 Molecule sizes: {min(n_nodes_count.keys())} to {max(n_nodes_count.keys())} atoms")
    
    # Create mappings
    atom_index = {z: i for i, z in enumerate(unique_atoms)}
    
    # Atomic number to element symbol mapping
    z_to_symbol = {
        1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
        14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 34: 'Se', 35: 'Br', 53: 'I'
    }
    
    atom_encoder = {z_to_symbol.get(z, f'Z{z}'): i for z, i in atom_index.items()}
    atom_decoder = [z_to_symbol.get(z, f'Z{z}') for z in unique_atoms]
    
    # Atom type counts (using the index mapping)
    atom_types = {atom_index[z]: count for z, count in atom_counts.items()}
    
    return {
        'atom_index': atom_index,
        'atom_encoder': atom_encoder, 
        'atom_decoder': atom_decoder,
        'n_nodes': dict(n_nodes_count),
        'max_n_nodes': max(n_nodes_count.keys()),
        'atom_types': atom_types,
        'total_molecules': total_molecules
    }

def format_python_dict(name, config, with_h=True):
    """Format configuration as a Python dictionary string."""
    if config is None:
        return f"# {name} = None  # Dataset not found\n"
    
    lines = [f"{name} = {{"]
    lines.append(f"    'name': '{name.replace('_with_h', '').replace('_without_h', '')}',")
    
    # atom_index
    atom_index_str = ", ".join(f"{k}: {v}" for k, v in config['atom_index'].items())
    lines.append(f"    'atom_index': {{{atom_index_str}}},")
    
    # atom_encoder
    encoder_items = [f"'{k}': {v}" for k, v in config['atom_encoder'].items()]
    encoder_str = ", ".join(encoder_items)
    lines.append(f"    'atom_encoder': {{{encoder_str}}},")
    
    # atom_decoder
    decoder_items = [f"'{item}'" for item in config['atom_decoder']]
    decoder_str = ", ".join(decoder_items)
    lines.append(f"    'atom_decoder': [{decoder_str}],")
    
    # n_nodes (formatted in multiple lines like QM40, sorted by n_nodes)
    n_nodes = dict(sorted(config['n_nodes'].items()))
    lines.append("    'n_nodes': {")
    
    # Group n_nodes by lines of ~10 items for readability
    items = list(n_nodes.items())
    for i in range(0, len(items), 10):
        chunk = items[i:i+10]
        chunk_str = ", ".join(f"{k}: {v}" for k, v in chunk)
        if i + 10 < len(items):
            lines.append(f"        {chunk_str},")
        else:
            lines.append(f"        {chunk_str}")
    
    lines.append("    },")
    
    # max_n_nodes
    lines.append(f"    'max_n_nodes': {config['max_n_nodes']},")
    
    # atom_types (sorted by key)
    atom_types_sorted = dict(sorted(config['atom_types'].items()))
    atom_types_str = ", ".join(f"{k}: {v}" for k, v in atom_types_sorted.items())
    lines.append(f"    'atom_types': {{{atom_types_str}}},")
    
    # with_h flag
    lines.append(f"    'with_h': {with_h}")
    
    lines.append("}")
    
    return "\n".join(lines)

def main():
    """Main function to analyze all datasets and generate configs."""
    
    # Output file
    output_file = "dataset_configs_output.txt"
    
    # Delete previous output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑️  Deleted previous output file: {output_file}")
    
    print(f"📄 Output will be saved to: {output_file}")
    
    # Open file for writing
    with open(output_file, 'w') as f:
        # Function to print to both console and file
        def print_both(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=f)  # Print to file
        
        print_both("🧬 Dataset Configuration Generator")
        print_both("=" * 60)
        
        # Define datasets to analyze (now analyzing all splits combined)
        datasets = [
            ('guacamol/with_h/raw', 'guacamol_with_h', True),
            ('guacamol/without_h/raw', 'guacamol_without_h', False),
            ('moses/with_h/raw', 'moses_with_h', True),
            ('moses/without_h/raw', 'moses_without_h', False),
        ]
        
        configs = {}
        
        # Analyze each dataset (all splits combined)
        for base_path, name, with_h in datasets:
            print_both(f"📊 Analyzing {name}...")
            config = analyze_npz_dataset_splits(base_path, name)
            configs[name] = (config, with_h)
        
        print_both("\n" + "=" * 60)
        print_both("📝 Generated Configuration Dictionaries")
        print_both("=" * 60)
        print_both()
        print_both("# Add these to configs/datasets_config.py:")
        print_both()
        
        # Output formatted configurations
        for name, (config, with_h) in configs.items():
            if config is not None:
                print_both(format_python_dict(name, config, with_h))
                print_both()
            else:
                print_both(f"# {name} configuration could not be generated (dataset not found)")
                print_both()
        
        print_both("=" * 60)
        print_both("✅ Configuration generation complete!")
        print_both()
        print_both("📋 Next steps:")
        print_both("1. Copy the above dictionaries to configs/datasets_config.py")
        print_both("2. Add imports if needed (they should already be compatible)")
        print_both("3. Test with your training scripts")
        print_both()
        print_both("💡 Note: This analysis was based on the test samples (~300 molecules per dataset).")
        print_both("   To get statistics for the FULL datasets, first process the complete datasets:")
        print_both("   - cd guacamol && python process_datasets.py (without --test flag)")
        print_both("   - cd moses && python process_datasets.py (without --test flag)")
        print_both("   Then re-run this script for production-quality configurations.")
    
    print(f"✅ Output saved to: {output_file}")

if __name__ == "__main__":
    main()
