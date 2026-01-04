#!/bin/bash
# Upload checkpoint to Hugging Face
# Run this script to upload your trained model checkpoint

set -e  # Exit on error

echo "🚀 EvoDiffMol Checkpoint Upload to Hugging Face"
echo "=================================================="
echo ""

# Configuration
HF_USERNAME="YOUR_USERNAME"  # TODO: Change this to your Hugging Face username
HF_REPO="EvoDiffMol"
CHECKPOINT_PATH="logs_moses/moses_without_h/moses_full_ddpm_2losses_2025_08_15__16_37_07_resume/checkpoints/80.pt"
CHECKPOINT_NAME="moses_without_h_80.pt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "💡 Please update CHECKPOINT_PATH in this script to point to your checkpoint file"
    exit 1
fi

# Get checkpoint size
CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
echo "📊 Checkpoint Details:"
echo "   Path: $CHECKPOINT_PATH"
echo "   Size: $CHECKPOINT_SIZE"
echo ""

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Check if logged in
echo "🔐 Checking Hugging Face authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please login:"
    echo "  1. Get your token at: https://huggingface.co/settings/tokens"
    echo "  2. Run: huggingface-cli login"
    echo ""
    exit 1
fi

HF_USER=$(huggingface-cli whoami | head -1)
echo "✅ Logged in as: $HF_USER"
echo ""

# Confirm upload
echo "📤 Ready to Upload:"
echo "   From: $CHECKPOINT_PATH"
echo "   To:   https://huggingface.co/$HF_USERNAME/$HF_REPO/$CHECKPOINT_NAME"
echo "   Size: $CHECKPOINT_SIZE"
echo ""
read -p "❓ Continue? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ Upload cancelled"
    exit 0
fi

echo ""
echo "⏫ Uploading checkpoint... (this may take several minutes)"
echo ""

# Upload using Python API (more reliable than CLI)
python3 << EOF
from huggingface_hub import HfApi
import sys

api = HfApi()

try:
    # Create repo if it doesn't exist
    print("📁 Creating/checking repository...")
    api.create_repo(
        repo_id="$HF_USERNAME/$HF_REPO",
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print("✅ Repository ready")
    print()
    
    # Upload file
    print("⏫ Uploading checkpoint...")
    api.upload_file(
        path_or_fileobj="$CHECKPOINT_PATH",
        path_in_repo="$CHECKPOINT_NAME",
        repo_id="$HF_USERNAME/$HF_REPO",
        repo_type="model"
    )
    
    print()
    print("=" * 60)
    print("✅ Upload Complete!")
    print("=" * 60)
    print()
    print("🔗 Your checkpoint is now available at:")
    print("   https://huggingface.co/$HF_USERNAME/$HF_REPO")
    print()
    print("📥 Direct download URL:")
    print("   https://huggingface.co/$HF_USERNAME/$HF_REPO/resolve/main/$CHECKPOINT_NAME")
    print()
    print("✅ Next Steps:")
    print("   1. Update assets/download_checkpoint.py (line 18):")
    print("      REPO_ID = '$HF_USERNAME/$HF_REPO'")
    print()
    print("   2. Update assets/README.md with your username")
    print()
    print("   3. Test download:")
    print("      python assets/download_checkpoint.py")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "🎉 Done!"

