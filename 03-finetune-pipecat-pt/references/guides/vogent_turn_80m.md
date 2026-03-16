---
title: "Vogent-Turn-80M Model Card"
source: https://huggingface.co/vogent/Vogent-Turn-80M
date: 2026-03-16
---

# Vogent-Turn-80M

A state-of-the-art multimodal turn detection model for voice AI systems, achieving **94.1% accuracy** by combining acoustic and linguistic signals for real-time conversational applications.

## Overview

| Attribute | Value |
|-----------|-------|
| **Developed by** | Vogent AI |
| **Model type** | Multimodal Turn Detection (Binary Classification) |
| **Language(s)** | English |
| **License** | Modified Apache-2.0 (horizontal voice agent platforms cannot set as default) |
| **Base model** | SmolLM2-135M (reduced to 80M parameters using first 12 layers) |
| **Model size** | 79.2M parameters (F32) |
| **Inference speed** | ~7ms on T4 GPU |

## Model Description

Vogent-Turn-80M determines when a speaker has finished their turn in a conversation by processing both:
- **Acoustic features** (via Whisper encoder)
- **Semantic context** (via language model)

This multimodal approach enables real-time, accurate predictions where audio-only or text-only methods fail.

### Resources

- **GitHub Repository:** https://github.com/vogent/vogent-turn
- **Technical Report:** https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents
- **Demo Space:** https://huggingface.co/spaces/vogent/vogent-turn-demo

## Quick Install

```bash
git clone https://github.com/vogent/vogent-turn.git
cd vogent-turn
pip install -e .
```

## Basic Usage

```python
from vogent_turn import TurnDetector
import soundfile as sf
import urllib.request

# Initialize detector
detector = TurnDetector(compile_model=True, warmup=True)

# Download and load audio
audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"
urllib.request.urlretrieve(audio_url, "sample.wav")
audio, sr = sf.read("sample.wav")

# Run turn detection with conversational context
result = detector.predict(
    audio,
    prev_line="What is your phone number",
    curr_line="My number is 804",
    sample_rate=sr,
    return_probs=True,
)

print(f"Turn complete: {result['is_endpoint']}")
print(f"Done speaking probability: {result['prob_endpoint']:.1%}")
```

## Available Interfaces

- **Python Library:** Direct integration with `TurnDetector` class
- **CLI Tool:** `vogent-turn-predict speech.wav --prev "What is your phone number" --curr "My number is 804"`

## Technical Architecture

### Model Architecture

**Components:**
1. **Audio Encoder:** Whisper-Tiny (processes up to 8 seconds of 16kHz audio)
2. **Text Model:** SmolLM-135M (12 layers, ~80M parameters)
3. **Multimodal Fusion:** Audio embeddings projected into LLM's input space
4. **Classifier:** Binary classification head (turn complete/incomplete)

### Processing Flow

```
1. Audio (16kHz PCM) -> Whisper Encoder -> Audio Embeddings (~400 tokens)
2. Text Context -> SmolLM Tokenizer -> Text Embeddings
3. Concatenate embeddings -> SmolLM Transformer -> Last token hidden state
4. Linear Classifier -> Softmax -> [P(continue), P(endpoint)]
```

## Training Details

### Preprocessing

- **Audio:** Last 8 seconds extracted via Whisper-Tiny encoder -> ~400 audio tokens
- **Text:** Full conversational context (assistant and user utterances)
- **Labels:** Binary classification (turn complete/incomplete)
- **Fusion:** Audio embeddings projected into LLM's input space and concatenated with text

### Training Hyperparameters

- **Training regime:** fp16 mixed precision
- **Base model:** SmolLM2-135M (first 12 layers)
- **Architecture:** Reduced from 135M to ~80M parameters via layer ablation

### Training Data

Diverse dataset combining:
- Human-collected conversational data
- Synthetic conversational data

## Evaluation Results

### Performance Metrics

- **Accuracy:** 94.1%
- **AUPRC:** 0.975

### Testing Data

Internal test set covering diverse conversational scenarios and edge cases where audio-only or text-only approaches fail.

## Compute Infrastructure

### Hardware Optimization

- `torch.compile` with max-autotune mode
- Dynamic tensor shapes without recompilation
- Pre-warmed bucket sizes (64, 128, 256, 512, 1024)

### Software

- **Framework:** PyTorch with torch.compile
- **Audio processing:** Whisper encoder (up to 8 seconds)

## Limitations

- **English-only** support; turn-taking conventions vary across languages and cultures
- **CPU inference** may be too slow for some real-time applications

## Citation

```bibtex
@misc{voturn2025,
  title={Vogent-Turn-80M: State-of-the-Art Turn Detection for Voice Agents},
  author={Varadarajan, Vignesh and Vytheeswaran, Jagath},
  year={2025},
  publisher={Vogent AI},
  howpublished={\url{https://huggingface.co/vogent/Vogent-Turn-80M}},
  note={Blog: \url{https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents}}
}
```

## Additional Resources

- **Full Documentation & Code:** https://github.com/vogent/vogent-turn
- **Platform:** https://vogent.ai
- **Enterprise Contact:** j@vogent.ai
- **Issues:** https://github.com/vogent/vogent-turn/issues

## Upcoming Releases

- Int8 quantized model for faster CPU deployment
- Multilingual versions
- Domain-specific adaptations
