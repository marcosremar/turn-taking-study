---
title: "Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection"
authors:
  - Koji Inoue
  - Bing'er Jiang
  - Erik Ekstedt
  - Tatsuya Kawahara
  - Gabriel Skantze
year: 2024
source: https://arxiv.org/abs/2401.04868
date_converted: 2026-03-16
---

## Abstract

A demonstration of a real-time and continuous turn-taking prediction system is presented. The system is based on a voice activity projection (VAP) model, which directly maps dialogue stereo audio to future voice activities. The VAP model includes contrastive predictive coding (CPC) and self-attention transformers, followed by a cross-attention transformer. The authors examine the effect of the input context audio length and demonstrate that the proposed system can operate in real-time with CPU settings, with minimal performance degradation.

## Key Contributions

1. **Real-time CPU inference**: Demonstrates that VAP can run in real-time on CPU by limiting transformer input context to ~1 second, with no accuracy loss.
2. **Continuous turn-taking prediction**: Unlike binary end-of-turn classifiers, VAP predicts future voice activity continuously at every time frame.
3. **Multilingual models**: Trained models for Japanese, English (Switchboard), and Mandarin Chinese (HKUST corpus).

## Architecture / Method Details

### VAP Model Architecture

The VAP model predicts future voice activity for both speakers in a dyadic dialogue from raw stereo audio:

1. **CPC Encoder** (Contrastive Predictive Coding): Pre-trained audio encoder processes each speaker's audio channel independently. Contains an auto-regressive GRU that builds representations over the full audio history (up to 20 seconds).

2. **Self-attention Transformers**: One layer per channel, processes CPC outputs. This is where the context length can be truncated for efficiency.

3. **Cross-attention Transformer**: 3 layers, captures interactive information between the two speaker channels.

4. **Output layers**: Linear layers for multitask learning:
   - **VAP objective**: Predicts joint voice activity of both speakers over the next 2 seconds.
   - **VAD subtask**: Voice activity detection auxiliary task.

### VAP Output Representation

- Predicts voice activities within a 2-second future window.
- Window divided into 4 binary bins: 0-200ms, 200-600ms, 600-1200ms, 1200-2000ms.
- Each bin is "voiced" or "unvoiced" for each speaker, yielding 256 possible activation states.
- Simplified into two metrics:
  - **p_now(s)**: Short-term prediction (0-600ms) -- "how likely is participant s to speak in the next 600ms"
  - **p_future(s)**: Longer-term prediction (600-2000ms)

### Model Configuration

- Self-attention: 1 layer per channel
- Cross-attention: 3 layers
- Attention heads: 4
- Unit size: 256
- Input: 50 frames per second

## Experimental Results

### Turn-Taking Prediction vs. Input Context Length

Evaluated on Japanese Travel Agency Task Dialogue dataset (92.5h training, 11.5h validation, 11.5h test). Test set: 1,023 turn transitions, 1,371 turn holds. Metric: balanced accuracy (random = 50%).

| Input Length (sec) | Balanced Accuracy (%) | Inference Time/Frame (ms) | Real-time Factor |
|---|---|---|---|
| 20.0 | 74.20 | 273.84 | 13.69 |
| 10.0 | 75.73 | 94.93 | 4.75 |
| 5.0 | 75.01 | 33.66 | 1.68 |
| 3.0 | 75.75 | 30.54 | 1.53 |
| **1.0** | **76.16** | **14.61** | **0.73** |
| 0.5 | 75.41 | 13.11 | 0.66 |
| 0.3 | 71.50 | 12.19 | 0.61 |
| 0.1 | 62.81 | 12.45 | 0.62 |

Key findings:
- **1-second context achieves the best accuracy (76.16%)** while running in real-time (factor 0.73).
- Performance degrades significantly below 0.3 seconds.
- The GRU in CPC retains long-term information, so the transformer only needs short context.
- CPU: Intel Xeon Gold 6128 @ 3.40 GHz.

### Multilingual Performance

English (Switchboard) and Mandarin Chinese (HKUST) models yielded similar results to Japanese, confirming the approach generalizes across languages.

## Relevance to Turn-Taking / End-of-Turn Detection

VAP is highly relevant to BabelCast's turn-taking needs:

1. **Continuous prediction**: Unlike threshold-based systems, VAP continuously predicts who will speak next. This is more informative than a binary end-of-turn decision -- we get probabilistic forecasts that can be used to adjust system responsiveness dynamically.

2. **Real-time CPU feasibility**: With 1-second context, VAP runs at 14.6ms per frame on CPU -- well within our 20ms frame budget. No GPU required for inference.

3. **Audio-only input**: VAP works directly on raw audio without requiring ASR transcription, which means zero additional latency from waiting for text. This is critical for our real-time translation pipeline where every millisecond of response delay matters.

4. **Stereo/dual-channel design**: VAP models the interaction between two speakers, which aligns with our meeting bot scenario where we need to track speaker turns in a conversation.

5. **76% balanced accuracy** on turn shift/hold prediction sets a baseline for what audio-only models can achieve. Combining with linguistic features (from our ASR) could push this higher.

6. **Open source**: Code available at github.com/ErikEkstedt/VoiceActivityProjection, making it practical to integrate or fine-tune for our domain.

7. **Limitation**: VAP was designed for dyadic (2-party) conversations. Multi-party meeting scenarios would require adaptation, though the model could still be applied to each speaker pair.
