#!/bin/bash
set -e

echo "=== Smart Turn Portuguese Fine-Tuning — Vast.ai ==="
echo "Started at: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"

cd /workspace

# Clone repo
if [ ! -d "turn-taking-study" ]; then
    git clone https://github.com/marcosremar/turn-taking-study.git
fi
cd turn-taking-study

# Install dependencies
echo "=== Installing dependencies ==="
pip install --no-cache-dir \
    torch torchaudio \
    transformers datasets huggingface_hub \
    soundfile numpy \
    onnx onnxruntime \
    2>&1 | tail -5

echo "=== Dependencies installed ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run fine-tuning
echo "=== Starting fine-tuning ==="
python finetune_smart_turn_gpu.py 2>&1 | tee /workspace/finetune.log

echo "=== DONE at $(date) ==="
echo "Results in /workspace/turn-taking-study/checkpoints/smart_turn_pt_v2/"

# Keep alive for result collection
sleep infinity
