---
title: "Announcing Smart Turn v3: CPU Inference in Just 12ms"
source: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
date_accessed: 2026-03-16
---

# Smart Turn v3: CPU Inference Breakthrough

## Overview

Daily announced Smart Turn v3, a dramatically improved voice turn detection model featuring unprecedented efficiency. The system achieves "CPU inference in just 12ms" while maintaining open-source accessibility for weights, training data, and scripts.

## Key Improvements

### Size and Performance
- **Model size:** Reduced to 8 MB (nearly 50x smaller than v2)
- **CPU inference speed:** 12ms on modern processors; 60ms on budget AWS instances
- **No GPU required:** Runs directly within Pipecat Cloud instances

### Language Coverage
Expanded to 23 languages including Arabic, Bengali, Chinese, Danish, Dutch, German, English, Finnish, French, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.

### Architecture
- **Foundation:** Whisper Tiny encoder (39M parameters)
- **Classification layers:** Adapted from Smart Turn v2
- **Total parameters:** 8M
- **Optimization:** int8 quantization via static QAT
- **Export format:** ONNX

## Competitive Comparison

| Metric | Smart Turn v3 | Krisp | Ultravox |
|--------|---------------|-------|----------|
| Size | 8 MB | 65 MB | 1.37 GB |
| Languages | 23 | English only | 26 |
| Availability | Open weights/data | Proprietary | Open weights |
| Focus | Single-inference latency | Decision confidence | Conversation context |

## Performance Benchmarks

CPU inference results (including preprocessing):

| Platform | Speed |
|----------|-------|
| AWS c7a.2xlarge | 12.6 ms |
| AWS c8g.2xlarge | 15.2 ms |
| Modal (6 cores) | 17.7 ms |
| AWS t3.2xlarge | 33.8 ms |
| AWS c8g.medium | 59.8 ms |
| AWS t3.medium | 94.8 ms |

GPU performance shows 3.3-6.6ms latency across various NVIDIA processors.

## Accuracy Results

Highest performers: Turkish (97.10%), Korean (96.85%), Japanese (96.76%)

Lower performers: Vietnamese (81.27%), Bengali (84.10%), Marathi (87.60%)

English achieved 94.31% accuracy across 2,846 test samples.

## Implementation

### With Pipecat
`LocalSmartTurnAnalyzerV3` integration available (v0.0.85+). Users download ONNX model from HuggingFace repository.

### Standalone
Direct ONNX runtime usage via provided inference scripts. Requires accompanying VAD model (Silero recommended) for optimal results.

## Resources

- Model weights on HuggingFace
- GitHub repository containing training code and inference examples
- Open test datasets available for benchmark reproduction
- Community dataset annotation available at smart-turn-dataset.pipecat.ai
