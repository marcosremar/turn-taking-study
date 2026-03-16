#!/bin/bash
# Start script for RunPod fine-tuning pod.
# Installs deps (cached in /workspace), then runs training with auto-resume.
# On pod restart, training resumes from the last checkpoint.

set -e

echo "=== Smart Turn v3 Fine-tuning Start Script ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Use /workspace for caching (persists across pod restarts)
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME=/workspace/huggingface
export TRANSFORMERS_CACHE=/workspace/huggingface
export TMPDIR=/workspace/tmp
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# Install Python deps (cached in /workspace so fast on restart)
echo "=== Installing dependencies ==="
pip install --quiet torch torchaudio transformers datasets numpy 2>&1 | tail -5

# Copy training script to /workspace if not already there
SCRIPT_DIR="/workspace/finetune"
mkdir -p "$SCRIPT_DIR"
if [ -f /finetune_smart_turn_v3.py ]; then
    cp /finetune_smart_turn_v3.py "$SCRIPT_DIR/"
elif [ -f /app/finetune_smart_turn_v3.py ]; then
    cp /app/finetune_smart_turn_v3.py "$SCRIPT_DIR/"
fi

# Check for existing checkpoint
if [ -f /workspace/checkpoints/smart_turn_pt_v3/resume_checkpoint.pt ]; then
    echo "=== Found resume checkpoint — will continue training ==="
fi

# Run training
echo "=== Starting training ==="
cd "$SCRIPT_DIR"
python finetune_smart_turn_v3.py 2>&1 | tee /workspace/training.log

echo "=== Training complete! ==="
echo "Results in /workspace/checkpoints/smart_turn_pt_v3/"
ls -la /workspace/checkpoints/smart_turn_pt_v3/

# Keep pod alive so we can download results
echo "=== Pod staying alive for result download. Use runpodctl to get files. ==="
sleep infinity
