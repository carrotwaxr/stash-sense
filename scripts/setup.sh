#!/bin/bash
# Setup script for stash-face-recognition

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$PROJECT_DIR/upstream-stashface"

echo "=== Stash Face Recognition Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check for NVIDIA GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Check for git-lfs
echo "Checking git-lfs..."
if command -v git-lfs &> /dev/null; then
    echo "git-lfs is installed"
else
    echo "git-lfs is NOT installed"
    echo "Install with: sudo apt-get install git-lfs"
    echo ""
fi

# Check if LFS files need pulling
echo ""
echo "Checking LFS files..."
cd "$UPSTREAM_DIR"
VOY_SIZE=$(stat -f%z data/face_arc.voy 2>/dev/null || stat -c%s data/face_arc.voy 2>/dev/null)
if [ "$VOY_SIZE" -lt 1000 ]; then
    echo "LFS files are still pointers (size: $VOY_SIZE bytes)"
    echo "Run: cd $UPSTREAM_DIR && git lfs pull"
else
    echo "LFS files appear to be downloaded (size: $VOY_SIZE bytes)"
fi

# Set up Python environment
echo ""
echo "Setting up Python environment..."
cd "$UPSTREAM_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies (this may take a while)..."
pip install --upgrade pip
pip install -r requirements.txt

# Test GPU in Python
echo ""
echo "Testing GPU access in Python..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Pull LFS files: cd $UPSTREAM_DIR && git lfs pull"
echo "2. Activate env:   source $UPSTREAM_DIR/.venv/bin/activate"
echo "3. Run stashface:  python app.py"
echo "4. Open browser:   http://localhost:7860"
