---
title: "Pipecat Smart Turn - Training Code (train.py)"
source: https://raw.githubusercontent.com/pipecat-ai/smart-turn/main/train.py
date: 2026-03-16
---

# SmartTurn V3 Speech Endpointing Model — Training Code Reference

> This document describes the training pipeline implemented in `train.py` from the
> [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) repository.
> The original file is Python source code; key architecture and configuration details
> are extracted below.

## Core Architecture

### SmartTurnV3Model

The model extends `WhisperPreTrainedModel` and uses Whisper's encoder as its backbone:

- **Input**: Log-mel spectrogram features with shape `(batch_size, 80, 800)`
- **Encoder**: Modified Whisper encoder with `max_source_positions=400`
- **Attention Pooling**: Neural network layer that learns weighted attention across time steps
- **Binary Classifier**: Multi-layer sequential network outputting probability scores

```
Input Features (batch, 80, 800)
    |
Whisper Encoder
    |
Attention Pooling [Linear -> Tanh -> Linear -> Softmax]
    |
Weighted Pooling (reduces to batch_size, hidden_size)
    |
Classifier [Linear -> LayerNorm -> GELU -> Dropout -> Linear -> GELU -> Linear]
    |
Sigmoid Output (probability of completion)
```

### Loss Function

Uses `BCEWithLogitsLoss` with dynamic positive sample weighting, clamped between 0.1 and 10.0 to handle class imbalance.

## Training Configuration

```
Base Model:                openai/whisper-tiny
Learning Rate:             5e-5
Epochs:                    4
Train Batch Size:          384
Eval Batch Size:           128
Warmup Ratio:              0.2
Weight Decay:              0.01
Eval/Save Steps:           500
Logging Steps:             100
LR Scheduler:              Cosine
Dataloader Workers:        6
```

## Datasets

**Training Sources**:
- `pipecat-ai/smart-turn-data-v3.2-train` (split 90/10 for train/eval)

**Test Sources**:
- `pipecat-ai/smart-turn-data-v3.2-test`

The system supports stratified analysis by language, midfiller presence, and endfiller presence.

## Data Pipeline

### OnDemandSmartTurnDataset

On-demand feature extraction pipeline:
- Audio truncation to last 8 seconds
- 16kHz sampling rate
- Padding to max length: `8 * 16000 = 128,000 samples`
- Normalization enabled

### SmartTurnDataCollator

Batches samples while preserving metadata (language, midfiller, endfiller flags) for downstream analysis.

## Export and Quantization

### ONNX FP32 Export

The wrapper ensures consistent output shapes across variable batch sizes:
- Dynamic batch dimension
- Fixed output shape: `(batch_size, 1)` for compatibility
- ONNX opset version 18
- Validation with batch sizes 1 and 2

### INT8 Static Quantization

**Configuration**:
- Calibration dataset size: 1024 samples
- Quantization format: QDQ (Quantize-Dequantize)
- Activation type: QUInt8
- Weight type: QInt8
- Per-channel quantization enabled
- Calibration method: Entropy
- Quantized operations: Conv, MatMul, Gemm

Process:
1. `quant_pre_process` for optimization and shape inference
2. Entropy-based calibration on training data subset
3. Static quantization with calibration reader

## Evaluation Metrics

Per-sample metrics computed during training and evaluation:

- Accuracy
- Precision (with zero_division warning handling)
- Recall
- F1 Score
- Confusion matrix components (TP, FP, TN, FN)
- Predicted positive/negative counts

### External Evaluation Callback

During training checkpoints, the system evaluates on all test splits and logs:
- Dataset-specific accuracies
- Language-stratified metrics
- Midfiller-stratified metrics
- Probability distributions (Weights & Biases histograms)
- Min/max/mean accuracy across categories
- Sample count distributions

## Key Utility Functions

- **`truncate_audio_to_last_n_seconds`**: Crops audio to preserve final n seconds (prevents padding inflation)
- **`process_predictions`**: Converts logits to probabilities and binary predictions using 0.5 threshold, with NaN validation
- **`compute_metrics`**: Scikit-learn based metric computation with detailed confusion matrix breakdown

## Workflows

### Training Run (`do_training_run`)

1. Initialize Weights & Biases project "speech-endpointing"
2. Load pretrained Whisper-tiny with custom head
3. Prepare datasets (load, split, wrap)
4. Train with specified hyperparameters
5. Save final model and feature extractor
6. Export to ONNX FP32
7. Return export path

### Quantization Run (`do_quantization_run`)

1. Load FP32 ONNX model
2. Create calibration dataset from training split
3. Apply pre-processing (optimization)
4. Execute static INT8 quantization
5. Return quantized model path

### Benchmark Run (`do_benchmark_run`)

1. Load feature extractor
2. Prepare test merged dataset
3. For each model path:
   - Create benchmark output directory
   - Run benchmark with batch size 256
   - Generate markdown report

## Dependencies

- PyTorch with ONNX export capabilities
- Transformers (HuggingFace) for Whisper and training
- ONNX Runtime with quantization support
- Scikit-learn for metrics
- Weights & Biases for experiment tracking
- torchcodec (runtime requirement for audio decoding)

## Error Handling

- Fast-fail check for missing `torchcodec` dependency
- ONNX model validation after export
- Non-finite value detection in predictions
- Exception logging for failed exports
