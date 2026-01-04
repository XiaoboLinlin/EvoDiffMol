#!/usr/bin/env python3
"""
Download pre-trained EvoDiffMol checkpoint from Hugging Face.

Usage:
    python assets/download_checkpoint.py
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ Error: huggingface_hub not installed")
    print("\n📦 Install it with:")
    print("   pip install huggingface-hub")
    sys.exit(1)

# Configuration
REPO_ID = "YOUR_USERNAME/EvoDiffMol"  # TODO: Update with your Hugging Face repo
CHECKPOINT_FILENAME = "moses_without_h_80.pt"
LOCAL_DIR = Path(__file__).parent / "checkpoints"

def download_checkpoint():
    """Download the pre-trained checkpoint from Hugging Face."""
    print("🚀 Downloading EvoDiffMol Pre-trained Checkpoint")
    print("=" * 60)
    print(f"📂 Repository: {REPO_ID}")
    print(f"📄 File: {CHECKPOINT_FILENAME}")
    print(f"💾 Destination: {LOCAL_DIR}")
    print()
    
    # Create local directory
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    local_path = LOCAL_DIR / CHECKPOINT_FILENAME
    if local_path.exists():
        print(f"✅ Checkpoint already exists: {local_path}")
        print(f"   File size: {local_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        response = input("\n❓ Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing checkpoint")
            return str(local_path)
    
    # Download
    try:
        print("⏬ Downloading... (this may take a few minutes)")
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILENAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        
        print()
        print("=" * 60)
        print("✅ Download Complete!")
        print(f"📁 Saved to: {downloaded_path}")
        print(f"💾 Size: {Path(downloaded_path).stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("🎉 You're ready to use EvoDiffMol!")
        print()
        print("💡 Quick Start:")
        print("   from evodiffmol import MoleculeGenerator")
        print(f"   gen = MoleculeGenerator(checkpoint_path='{downloaded_path}')")
        
        return downloaded_path
        
    except Exception as e:
        print(f"\n❌ Error downloading checkpoint: {e}")
        print("\n💡 Manual Download:")
        print(f"   1. Visit: https://huggingface.co/{REPO_ID}")
        print(f"   2. Download: {CHECKPOINT_FILENAME}")
        print(f"   3. Save to: {LOCAL_DIR}")
        sys.exit(1)

if __name__ == "__main__":
    download_checkpoint()

