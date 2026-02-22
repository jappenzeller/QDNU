#!/bin/bash
# EC2 Instance Bootstrap Script for QDNU Parallel LOSO
# D-002: Set up c5.4xlarge (16 vCPU, 32GB RAM) spot instance
#
# Usage:
#   1. Launch Ubuntu 22.04 AMI on c5.4xlarge spot instance
#   2. SSH into instance and run: bash ec2_setup.sh
#   3. Run parallel LOSO: python scripts/ec2_parallel_loso.py
#
# Prerequisites:
#   - EBS volume mounted at /data with CHB-MIT dataset
#   - Instance role with S3 read access (optional, for dataset download)

set -e

echo "=============================================="
echo "QDNU EC2 Setup Script (D-002)"
echo "=============================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv htop tmux

# Create project directory
echo "[2/6] Setting up project directory..."
PROJECT_DIR="/home/ubuntu/QDNU"
if [ ! -d "$PROJECT_DIR" ]; then
    git clone https://github.com/jappenzeller/QDNU.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip -q

# Core ML dependencies
pip install -q \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    xgboost>=2.0.0 \
    pyriemann>=0.4.0 \
    joblib>=1.3.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0

# EEG processing
pip install -q pyedflib --no-deps

# Optional: Qiskit for quantum simulations (comment out if not needed)
# pip install -q qiskit>=1.0.0 qiskit-aer>=0.13.0

echo "[5/6] Verifying installation..."
python3 -c "import numpy, scipy, sklearn, xgboost, joblib, pyedflib; print('All dependencies OK')"

# Data directory setup
echo "[6/6] Setting up data directory..."
DATA_DIR="/data/chbmit"
if [ -d "$DATA_DIR" ]; then
    echo "CHB-MIT data found at $DATA_DIR"
    # Create symlink if needed
    mkdir -p "$(dirname "$PROJECT_DIR/data/chbmit")"
    ln -sf "$DATA_DIR" "$PROJECT_DIR/data/chbmit" 2>/dev/null || true
else
    echo "WARNING: CHB-MIT data not found at $DATA_DIR"
    echo "Please mount EBS volume with dataset or download manually"
    echo ""
    echo "To download CHB-MIT dataset (~5GB):"
    echo "  mkdir -p /data/chbmit"
    echo "  wget -r -np -nH --cut-dirs=3 -P /data/chbmit \\"
    echo "    https://physionet.org/files/chbmit/1.0.0/"
fi

# Create results directory
mkdir -p "$PROJECT_DIR/results/per_patient"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To run parallel LOSO:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python scripts/ec2_parallel_loso.py"
echo ""
echo "Monitor with:"
echo "  htop              # CPU/memory usage"
echo "  tail -f /tmp/loso.log  # Progress log"
echo ""
echo "Recommended: Run in tmux session:"
echo "  tmux new -s loso"
echo "  python scripts/ec2_parallel_loso.py"
echo "  # Ctrl+B, D to detach"
