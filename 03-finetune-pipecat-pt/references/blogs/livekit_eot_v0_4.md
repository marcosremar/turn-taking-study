---
title: "Improved End-of-Turn Model Cuts Voice AI Interruptions 39%"
source: https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/
date_accessed: 2026-03-16
---

# Improved End-of-Turn Model Cuts Voice AI Interruptions 39%

## Overview

LiveKit released an updated transformer-based end-of-turn detection model, `v0.4.1-intl`, featuring "a 39.23% relative reduction in false-positive interruptions" compared to the previous version. The model now emphasizes structured data handling and multilingual performance across supported languages.

## Key Improvements

### Performance Metrics

The benchmark demonstrates consistent gains across all tested languages:

| Language | v0.3.0 Error Rate | v0.4.1 Error Rate | Relative Improvement |
|----------|------------------|------------------|----------------------|
| Chinese | 18.70% | 13.40% | 28.34% |
| Dutch | 26.10% | 11.90% | 54.33% |
| English | 16.60% | 13.00% | 21.69% |
| French | 16.80% | 11.10% | 33.93% |
| German | 23.40% | 12.20% | 47.86% |
| Hindi | 5.40% | 3.70% | 31.48% |
| Indonesian | 17.00% | 10.60% | 37.65% |
| Italian | 20.10% | 14.90% | 25.87% |
| Japanese | 19.70% | 11.20% | 43.15% |
| Korean | 7.90% | 5.50% | 30.38% |
| Portuguese | 23.30% | 12.60% | 45.97% |
| Russian | 19.50% | 12.00% | 38.46% |
| Spanish | 21.50% | 14.00% | 33.88% |
| Turkish | 25.40% | 12.70% | 50.0% |
| **All** | **18.66%** | **11.34%** | **39.23%** |

Error rate represents the false positive rate at a fixed true positive rate of 99.3%.

## Technical Architecture

### The Challenge of End-of-Turn Detection

Voice AI systems must detect speech completion using three primary cues:

- **Semantic content**: Word meaning and language understanding
- **Context**: Dialogue history and conversational flow
- **Prosody**: Tone, pauses, and rhythmic patterns

The model uses an LLM backbone to effectively combine content and context information.

### Structured Data Handling

A primary enhancement addresses structured information collection. When users provide phone numbers, emails, or addresses, natural speech markers (intonation changes, grammatical endings) are absent. The updated model "addresses this by inferring expected formats from the agent's prompt," enabling:

- Detection of complete phone numbers (typically 10 digits for US numbers)
- Email pattern recognition
- Address validation including street, city, state, and zip code

The visualization tool demonstrates the improvement: `v0.3.0-intl` incorrectly detected end points after individual digits, while `v0.4.1-intl` waited for the complete sequence.

### Handling Speech-to-Text Variability

Training data incorporated multiple STT output formats. One engine might transcribe "forty two" as words while another outputs "42" numerically. By training across "common STT output formats," the model maintains consistent performance across different provider implementations.

### Multilingual Generalization

Despite structured data enhancements targeting English training data, improvements transferred across languages -- particularly for phone number and address detection in Spanish, French, and other languages. This stems from the Qwen2.5 base model's multilingual pretraining, which encodes "knowledge of global formats."

## Model Architecture & Optimization

### Base Model Selection

The team selected **Qwen2.5-0.5B-Instruct** for its balance of performance and low-latency CPU inference capabilities.

### Knowledge Distillation

To improve multilingual accuracy without sacrificing efficiency:

1. A larger **Qwen2.5-7B-Instruct** teacher model was trained
2. Knowledge was distilled into the smaller 0.5B student model
3. The distilled model achieves "higher multilingual accuracy with the efficiency of the smaller size"

Training curves show the distilled model outperforming the baseline 0.5B version and approaching teacher performance after approximately 1,500 steps.

## Availability & Deprecation

- **Deployment**: Available in Agents Python 1.3.0 and Agents JS 1.0.19
- **Model**: `MultilingualModel` now recommended for all use cases
- **Deprecation**: The legacy `EnglishModel` is being deprecated as the multilingual version "not only matches but in most cases exceeds the performance"

## Observability Integration

LiveKit Agents now include built-in observability for end-of-turn detection. When Agent Observability is enabled, "every turn detection decision is logged with the exact input the model saw." Production debugging accesses `eou_detection` traces showing full prediction context.

## Future Directions

Current implementation relies on transcribed text. Future iterations will integrate raw audio features like pauses and emphasis through multimodal architectures, "fusing prosody directly into predictions for more precise detection."

---

**Resources**:
- [GitHub Repository](https://github.com/livekit/agents)
- [Visualization Tool](https://huggingface.co/spaces/livekit/eot-visualization)
- [Voice AI Quickstart](https://docs.livekit.io/agents/start/voice-ai/)
