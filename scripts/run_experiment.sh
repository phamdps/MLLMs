#!/usr/bin/env bash

# ==============================================================================
# Master Experiment Runner: MLLM-Powered Transportation Digital Twin
# ==============================================================================

set -e  # Exit immediately if any command exits with a non-zero status

CONFIG_PATH="config/config.yaml"
CHECKPOINT_DIR="checkpoints"

echo "========================================================"
echo "🚀 Starting MLLM Transportation Digital Twin Pipeline"
echo "========================================================"

# Step 1: Verify Environment & GPU Status
echo "--- [1/5] Checking GPU & Torch Environment ---"
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"

# Step 2: Ensure Required Directories Exist
echo "--- [2/5] Initializing Directory Structure ---"
mkdir -p data/raw/METR-LA data/processed checkpoints logs

# Step 3: Check Raw Datasets (Download if missing)
if [ ! -f "data/raw/METR-LA/metr_la.h5" ] || [ ! -f "data/raw/METR-LA/adj_mx.pkl" ]; then
    echo "--- [3/5] Raw METR-LA dataset files not found. Downloading via scripts/download_data.py ---"
    python3 scripts/download_data.py
else
    echo "--- [3/5] METR-LA dataset files found. Skipping download. ---"
fi

# Step 4: Run Multi-Task Training Pipeline
echo "--- [4/5] Executing Multimodal Model Training ---"
python3 train/train_model.py --config "$CONFIG_PATH"

# Step 5: Verification and Checkpoint Notice
echo "--- [5/5] Experiment Complete! ---"
if [ -f "$CHECKPOINT_DIR/best_transportation_mllm.pt" ]; then
    echo "✅ Success! Best model checkpoint saved to: $CHECKPOINT_DIR/best_transportation_mllm.pt"
else
    echo "⚠️ Warning: Training finished, but checkpoint file was not found."
fi

echo "========================================================"
echo "🎉 All pipeline stages executed successfully!"
echo "========================================================"