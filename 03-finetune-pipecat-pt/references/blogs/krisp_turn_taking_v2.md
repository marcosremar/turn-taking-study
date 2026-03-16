---
title: "Krisp Turn-Taking v2 - Voice AI VIVA SDK"
source: https://krisp.ai/blog/krisp-turn-taking-v2-voice-ai-viva-sdk/
date_accessed: 2026-03-16
---

# Audio-Only Turn-Taking Model v2 - Krisp

## Overview

Krisp has released Turn-Taking v2, an updated model for detecting end-of-turns in real-time conversational AI systems. The model processes audio input only, making it suitable for "human-bot interactions" and other voice AI applications integrated through Krisp's VIVA SDK.

## Technical Architecture

The latest iteration, **krisp-viva-tt-v2**, represents a substantial advancement over its predecessor. According to the engineering team, it was "trained on a more diverse and better-structured dataset, with richer data augmentations that help the model perform more reliably in real-world conditions."

## Key Improvements

- Enhanced robustness in noisy environments
- Superior accuracy when combined with Krisp's Voice Isolation models
- Faster and more stable turn detection during live conversations

## Performance Benchmarks

### Clean Audio Testing

Testing evaluated approximately 1,800 real conversation samples (~1,000 "hold" cases and ~800 "shift" cases) with mild background noise:

| Model | Balanced Accuracy | AUC | F1 Score |
|-------|-------------------|-----|----------|
| krisp-viva-tt-v1 | 0.82 | 0.89 | 0.804 |
| **krisp-viva-tt-v2** | **0.823** | **0.904** | **0.813** |

### Noisy Audio Testing (5-15 dB noise levels)

| Model | Balanced Accuracy | AUC | F1 Score |
|-------|-------------------|-----|----------|
| krisp-viva-tt-v1 | 0.723 | 0.799 | 0.71 |
| **krisp-viva-tt-v2** | **0.768** | **0.842** | **0.757** |

V2 demonstrated "up to a 6% improvement in F1 score under noisy conditions."

### Post-Processing with Voice Isolation

After applying background noise and voice removal through krisp-viva-tel-v2:

| Model | Balanced Accuracy | AUC | F1 Score |
|-------|-------------------|-----|----------|
| krisp-viva-tt-v1 | 0.787 | 0.854 | 0.775 |
| **krisp-viva-tt-v2** | **0.816** | **0.885** | **0.808** |

## Availability

The model is now available as part of Krisp's VIVA SDK, designed for developers building Voice AI agents and conversational systems.
