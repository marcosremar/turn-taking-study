---
title: "LiveKit Turn Detector Model Card"
source: https://huggingface.co/livekit/turn-detector
date: 2026-03-16
---

# LiveKit Turn Detector

An open-weights language model for contextually-aware end-of-utterance (EOU) detection in voice AI applications. The model predicts whether a user has finished speaking based on the semantic content of their transcribed speech, providing a critical complement to voice activity detection (VAD) systems.

> For installation, usage examples, and integration guides, see the [LiveKit documentation](https://docs.livekit.io/agents/logic/turns/turn-detector/).

## Overview

Traditional voice agents rely on voice activity detection (VAD) to determine when a user has finished speaking. VAD works by detecting the presence or absence of speech in an audio signal and applying a silence timer. While effective for detecting pauses, VAD lacks language understanding and frequently causes false positives.

**Example:** A user who says *"I need to think about that for a moment..."* and then pauses will be interrupted by a VAD-only system, even though they clearly intend to continue.

This model adds semantic understanding to the turn detection process by:
- Analyzing transcribed text of conversations in real time
- Predicting the probability that the user has completed their turn
- Reducing unwanted interruptions when integrated with VAD
- Handling structured data input effectively (addresses, phone numbers, email addresses, credit card numbers)

## Model Variants

**Multilingual** (recommended) and **English-only** (deprecated) are distributed as INT8 quantized ONNX models (`model_q8.onnx`) optimized for CPU inference.

> **The English-only model (`EnglishModel`) is deprecated.** Use the **multilingual model (`MultilingualModel`)** for all new projects. The multilingual model provides better accuracy across all languages thanks to knowledge distillation from a larger teacher model and expanded training dataset.

## How It Works

The model operates on transcribed text from a speech-to-text (STT) system, not raw audio.

### Process Flow

1. **Input**: Recent conversation history (up to **6 turns**, truncated to **128 tokens**) is formatted using the [Qwen chat template](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with `<|im_start|>` / `<|im_end|>` delimiters. The final user message is left _without_ the closing `<|im_end|>` token.

2. **Prediction**: The model predicts the probability of the `<|im_end|>` token appearing next:
   - **High probability** -> user has likely finished their utterance
   - **Low probability** -> user is likely to continue

3. **Thresholding**: Per-language thresholds (stored in `languages.json`) convert raw probability into a binary decision, tuned to balance responsiveness and accuracy for each supported language.

4. **Integration with VAD**: Works alongside the [Silero VAD](https://docs.livekit.io/agents/logic/turns/vad/) plugin. VAD handles speech presence detection and interruption triggering, while this model provides the semantic signal for when to commit a turn.

### Text Preprocessing

**Multilingual variant** applies:
- NFKC unicode normalization
- Lowercasing
- Punctuation removal (preserving apostrophes and hyphens)
- Whitespace collapsing

**English-only variant** passes raw transcribed text without normalization.

## Architecture and Training

### Base Model

Both variants are fine-tuned from [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), selected for strong performance on this task while enabling low-latency CPU inference.

**Model Specifications:**
- **Parameters**: 0.1B
- **Tensor Type**: F32
- **Chat Template**: Qwen format

### Knowledge Distillation

A **Qwen2.5-7B-Instruct** model was first fine-tuned as a teacher on end-of-turn prediction. Its knowledge was then distilled into the 0.5B student model:
- Distilled model approaches teacher-level accuracy
- Maintains efficiency of smaller architecture
- Converges after approximately 1,500 training steps

### Training Data

The training dataset is a mix of:

- **Real call center transcripts** covering diverse conversational patterns
- **Synthetic dialogues** emphasizing structured data input:
  - Addresses
  - Email addresses
  - Phone numbers
  - Credit card numbers
- **Multi-format STT outputs** to handle provider variation (e.g., "forty two" vs. "42"), ensuring consistent predictions across different STT engines without runtime overhead

*Note: Structured data enhancements were added only to the English training set, but performance improvements generalized across languages due to the multilingual knowledge encoded in Qwen2.5.*

### Quantization

The trained model is exported to ONNX format and quantized to INT8 (`model_q8.onnx`), enabling efficient CPU-only inference with ONNX Runtime.

## Supported Languages

The multilingual model supports **14 languages**. The model relies on the STT provider to report the detected language, which is then used to select the appropriate per-language threshold.

**Supported:** English, Spanish, French, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Turkish, Russian, Hindi

## Benchmarks

### Detection Accuracy (Multilingual Variant)

- **True positive** -- the model correctly identifies the user has finished speaking
- **True negative** -- the model correctly identifies the user will continue speaking

| Language | True Positive Rate | True Negative Rate |
|----------|-------------------|-------------------|
| Hindi | 99.4% | 96.3% |
| Korean | 99.3% | 94.5% |
| French | 99.3% | 88.9% |
| Indonesian | 99.3% | 89.4% |
| Japanese | 99.3% | 88.8% |
| Dutch | 99.3% | 88.1% |
| Russian | 99.3% | 88.0% |
| German | 99.3% | 87.8% |
| Portuguese | 99.4% | 87.4% |
| Turkish | 99.3% | 87.3% |
| English | 99.3% | 87.0% |
| Chinese | 99.3% | 86.6% |
| Spanish | 99.3% | 86.0% |
| Italian | 99.3% | 85.1% |

### Improvement Over Prior Version

The multilingual v0.4.1 release achieved a **39.23% relative improvement** in handling structured inputs (emails, addresses, phone numbers, credit card numbers) compared to the prior version, reducing premature interruptions during data collection scenarios.

## Usage

The model is designed for use as a turn detection plugin within the [LiveKit Agents](https://github.com/livekit/agents) framework.

For complete installation instructions, code examples (Python and Node.js), and configuration options, see the **[LiveKit turn detector plugin documentation](https://docs.livekit.io/agents/logic/turns/turn-detector/)**.

For broader context on how turn detection fits into the voice pipeline -- including VAD configuration, interruption handling, and manual turn control -- see the **[Turns overview](https://docs.livekit.io/agents/logic/turns/)**.

## Deployment Requirements

- **Runtime**: CPU-only (no GPU required). Uses [ONNX Runtime](https://onnxruntime.ai/) with the `CPUExecutionProvider`.
- **RAM**: <500 MB for the multilingual model
- **Instance type**: Use compute-optimized instances (e.g., AWS c6i, c7i). Avoid burstable instances (e.g., AWS t3, t4g) to prevent inference timeouts from CPU credit exhaustion.
- **LiveKit Cloud**: The model is deployed globally on LiveKit Cloud. Agents running there automatically use the optimized remote inference service with no local resource requirements.

## Limitations

- **Text-only input**: The model operates on STT-transcribed text and cannot incorporate prosodic cues such as pauses, intonation, or emphasis. Future versions may integrate multimodal audio features.
- **STT dependency**: Prediction quality depends on the accuracy and output format of the upstream STT provider. Mismatches between training and deployment STT formats may degrade performance.
- **Context window**: Limited to 128 tokens across a maximum of 6 conversation turns.
- **Language coverage**: Currently supports 14 languages. Performance on unsupported languages is undefined.
- **Realtime model compatibility**: Cannot be used with audio-native realtime models (e.g., OpenAI Realtime API) without adding a separate STT service, which incurs additional cost and latency.

## License

This model is released under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).

## Resources

- **[Documentation](https://docs.livekit.io/agents/logic/turns/turn-detector/)**: Full plugin documentation, installation, and integration guide
- **[Turns Overview](https://docs.livekit.io/agents/logic/turns/)**: How turn detection fits into the LiveKit Agents voice pipeline
- **[Blog: Improved End-of-Turn Model](https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/)**: Technical deep dive on multilingual distillation approach and benchmarks
- **[Blog: Using a Transformer for Turn Detection](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection/)**: Original blog post introducing the concept and architecture
- **[Video: LiveKit Turn Detector](https://youtu.be/OZG0oZKctgw)**: Overview video demonstrating the plugin
- **[GitHub: Plugin Source](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-turn-detector)**: Source code for the `livekit-plugins-turn-detector` package
- **[PyPI](https://pypi.org/project/livekit-plugins-turn-detector/)** | **[npm](https://www.npmjs.com/package/@livekit/agents-plugin-livekit)**: Package registries

---

**Model Stats:**
- Downloads last month: 240,507
- Model size: 0.1B params
- Format: Safetensors (INT8 quantized ONNX)
- Base Model: Qwen/Qwen2.5-0.5B-Instruct
