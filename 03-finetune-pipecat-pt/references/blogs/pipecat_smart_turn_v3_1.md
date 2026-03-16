---
title: "Improved Accuracy in Smart Turn v3.1"
source: https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/
date_accessed: 2026-03-16
---

# Improved Accuracy in Smart Turn v3.1

Smart Turn v3.1 represents a significant advancement in conversation turn detection, leveraging expanded training datasets to enhance model performance across languages.

## Overview

Smart Turn v3.1 maintains the same architecture as its predecessor while offering improved accuracy through additional human audio training data and refined quantization methods. The update functions as a direct replacement for v3.0, requiring no code modifications to existing implementations.

## Training Data Enhancement

The model benefits from contributions by three specialized data partners:

**Liva AI** provides "real human voice data to improve speech models" across multiple languages and dialects, founded by researchers with publications in IEEE and machine learning conferences.

**Midcentury** develops "multimodal-native research" datasets spanning 12+ languages, emphasizing real-world performance scenarios.

**MundoAI** constructs "the world's largest and highest quality multimodal datasets" across 16+ languages, prioritizing multilingual diversity.

These partners contributed audio samples in English and Spanish, publicly released through HuggingFace as `smart-turn-data-v3.1-train` and `smart-turn-data-v3.1-test` datasets.

## Model Variants

Two deployment options accommodate different hardware configurations:

- **CPU Model (8MB, int8 quantized)**: Lightweight variant enabling ~12ms CPU inference, matching v3.0 sizing
- **GPU Model (32MB, unquantized)**: Enhanced variant for GPU deployment with ~1% accuracy improvement

## Accuracy Improvements

Testing on the new v3.1 dataset demonstrates substantial gains:

| Language | v3.0 | v3.1 (8MB) | v3.1 (32MB) |
|----------|------|-----------|-----------|
| English | 88.3% | 94.7% | 95.6% |
| Spanish | 86.7% | 90.1% | 91.0% |

The model supports 23 total languages; the remaining 21 maintain parity with v3.0 performance.

## Performance Benchmarks

Single-inference latency across representative hardware:

| Device | v3.1 (8MB) | v3.1 (32MB) | Preprocessing |
|--------|-----------|-----------|---------------|
| GPU (NVIDIA L40S) | 2 ms | 1 ms | 1 ms |
| GPU (NVIDIA T4) | 5 ms | 4 ms | 2 ms |
| CPU (AWS c7a.2xlarge) | 9 ms | 13 ms | 7 ms |
| CPU (AWS c8g.2xlarge) | 20 ms | 32 ms | 9 ms |
| CPU (AWS c7a.medium) | 37 ms | 73 ms | 7 ms |
| CPU (AWS c8g.medium) | 57 ms | 159 ms | 9 ms |

Performance optimization is achievable through environment variable configuration:

```
OMP_NUM_THREADS=1
OMP_WAIT_POLICY="PASSIVE"
```

## Resources

- Model weights: https://huggingface.co/pipecat-ai/smart-turn-v3
- GitHub repository: https://github.com/pipecat-ai/smart-turn
- Datasets: https://huggingface.co/pipecat-ai
