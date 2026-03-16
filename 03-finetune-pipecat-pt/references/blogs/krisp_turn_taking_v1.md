---
title: "Turn-Taking for Voice AI"
source: https://krisp.ai/blog/turn-taking-for-voice-ai/
date_accessed: 2026-03-16
---

# Audio-only, 6M Weights Turn-Taking Model for Voice AI Agents

## Overview

Krisp has developed a lightweight turn-taking model designed to determine when speakers should transition in voice conversations. This technology is included at no additional cost in Krisp's VIVA SDK.

## What is Turn-Taking?

Turn-taking represents "the fundamental mechanism by which participants in a conversation coordinate who speaks when." In voice AI contexts, it manages when agents should listen, speak, or remain silent. The model addresses two primary tasks:

- **End-of-turn prediction**: Detecting when the current speaker will finish
- **Backchannel prediction**: Recognizing brief acknowledgments like "uh-huh" without speaker transitions

## Implementation Approaches

**Audio-based methods** analyze acoustic features including pitch variations, energy levels, intonation, pauses, and speaking rate. These enable low-latency responses critical for real-time scenarios.

**Text-based approaches** examine transcribed content, identifying linguistic cues like sentence boundaries and discourse markers, though they typically require larger architectures.

**Multimodal (fusion) systems** combine both modalities, leveraging acoustic cues alongside semantic understanding.

## Key Challenges

- **Hesitation vs. completion**: Distinguishing filler words ("um," "you know") from true turn endings
- **Natural pauses**: Differentiating conversational pauses from actual turn boundaries
- **Response speed**: Minimizing latency while maintaining accuracy
- **Speaking diversity**: Accounting for varying rhythms, accents, and intonation patterns

## Model Architecture

The Krisp model processes 100ms audio frames, outputting confidence scores (0-1) indicating shift likelihood. A configurable threshold determines binary shift predictions, with a default 5-second maximum hold duration.

## Performance Comparison

| Attribute | Krisp TT | SmartTurn v1 | SmartTurn v2 | VAD-based |
|-----------|----------|-------------|-------------|-----------|
| Parameters | 6.1M | 581M | 95M | 260k |
| Model Size | 65 MB | 2.3 GB | 360 MB | 2.3 MB |
| Execution | CPU | GPU | GPU | CPU |

## Evaluation Metrics

### Accuracy Results

| Model | Balanced Accuracy | AUC Shift | F1 Score Shift | F1 Score Hold | AUC (MST vs FPR) |
|-------|------------------|-----------|----------------|---------------|-----------------|
| Krisp TT | 0.82 | 0.89 | 0.80 | 0.83 | 0.21 |
| VAD-based | 0.59 | -- | 0.48 | 0.70 | -- |
| SmartTurn V1 | 0.78 | 0.86 | 0.73 | 0.84 | 0.39 |
| SmartTurn V2 | 0.78 | 0.83 | 0.76 | 0.78 | 0.44 |

### Training Data

- Approximately 2,000 hours of conversational speech
- Around 700,000 speaker turns
- Test dataset: 1,875 manually labeled audio samples

## Key Performance Findings

The Krisp model achieves "considerably faster average response time (0.9 vs. 1.3 seconds at a 0.06 FPR) compared to SmartTurn" while being 5-10 times smaller and optimized for CPU execution.

## Future Development

**Planned enhancements include:**

- Text-based turn-taking using custom neural networks
- Multimodal audio-text fusion for improved accuracy
- Backchannel detection to distinguish meaningful interruptions from casual listening acknowledgments

---

*Article published August 5, 2025 by Krisp Engineering Team*
