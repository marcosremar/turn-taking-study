---
title: "Speculative End-Turn Detector for Efficient Speech Chatbot Assistant"
authors:
  - Hyunjong Ok
  - Suho Yoo
  - Jaeho Lee
year: 2025
source: https://arxiv.org/abs/2503.23439
date_converted: 2026-03-16
---

## Abstract

Spoken dialogue systems powered by large language models have demonstrated remarkable abilities in understanding human speech and generating appropriate spoken responses. However, these systems struggle with end-turn detection (ETD) -- the ability to distinguish between user turn completion and hesitation. This limitation often leads to premature or delayed responses. The authors introduce the ETD Dataset, the first public dataset for end-turn detection, consisting of both synthetic speech data (generated with TTS) and real-world speech data collected from web sources (120k+ samples, 300+ hours). They propose SpeculativeETD, a collaborative inference framework that balances efficiency and accuracy: a lightweight GRU-based model (1M params) rapidly detects non-speaking units on local devices, while a high-performance Wav2vec-based model (94M params) runs on the server to classify turn ends vs. pauses. Experiments demonstrate significantly improved ETD accuracy while keeping computation low.

## Key Contributions

1. **ETD Dataset**: First open-source dataset specifically for end-turn detection, with 120k+ samples and 300+ hours of conversational data (synthetic + real).
2. **SpeculativeETD**: Novel collaborative inference framework combining a lightweight on-device model with a server-side model for efficient real-time end-turn detection.
3. **Three-state formulation**: Formally defines ETD as a ternary classification -- Speaking Unit (SU), Pause (within-turn silence), and Gap (end-of-turn silence).

## Architecture / Method Details

### ETD Task Formulation

At each time t, the speaker is in one of three states:
- **Speaking Unit (SU)**: Speaker is actively in speech.
- **Pause**: Speaker is not in speech but intends to continue (within-turn).
- **Gap**: Speaker has finished talking, marking end of turn.

### ETD Dataset Construction

**Synthetic data** (from MultiWOZ corpus + TTS):
- **Base variant (V1)**: Direct TTS of text dialogues -- only SU and Gap states.
- **w/ Pause variant (V2)**: Randomly extend TTS hesitations into pauses of 1.5-3.0 seconds.
- **w/ Filler words variant (V3)**: Inject filler words ("um", "uh", "hmm", etc.) at random locations, add pauses after them.
- TTS models: MeloTTS and Google Cloud TTS for diversity.

| Split | Samples | Duration (h) | Avg Duration (s) |
|-------|---------|---------------|-------------------|
| Train | 96,773 | 158.4 | 5.89 |
| Dev | 12,840 | 21.25 | 5.96 |
| Test | 12,868 | 21.21 | 5.93 |
| **Total** | **122,481** | **200.86** | **5.90** |

**Real data** (YouTube + Buckeye speech corpus):
- Speaker diarization to ensure exactly 2 speakers per sample.
- Silences >200ms labeled as Pause (same speaker) or Gap (different speaker).
- Language filtering with Whisper (English only, 99.07% accuracy).
- Total: 8,022 samples, 115.08 hours.

### SpeculativeETD Framework

Two-stage collaborative inference:

1. **On-device (GRU, 1M params)**: Processes streaming audio frame-by-frame. Performs binary classification: Speaking Unit vs. non-SU (Gap or Pause). This is a simpler task achievable with tiny models.

2. **Server-side (Wav2vec 2.0, 94M params)**: Only invoked when the GRU detects silence. Receives the speech segment and classifies: Gap vs. Pause. Only runs once per silence segment, not every frame.

Key advantages:
- The expensive Gap-vs-Pause decision happens only once per silence segment, not at every frame (10x+ computation savings).
- On-device GRU handles continuous streaming with sub-millisecond latency.
- Communication between device and server is infrequent (once per silence, not continuous).

### Models Evaluated

- **VAP** (Ekstedt & Skantze, 2022): Pretrained turn-taking model with frozen encoder, predictor trained on ETD dataset.
- **GRU** (1M params): 2 Conv2D layers + 1 GRU layer, trained from scratch.
- **Wav2vec 2.0** (94M params): Full fine-tuning on ETD dataset.
- **SpeculativeETD**: GRU (on-device) + Wav2vec 2.0 (server-side).

## Experimental Results

### Binary Classification (Gap vs. Pause)

| Method | Params | Synthetic Acc. | Real Acc. |
|--------|--------|---------------|-----------|
| VAP | - | 86.2 | 57.2 |
| GRU | 1M | 79.3 | 48.3 |
| Wav2vec 2.0 | 94M | **99.5** | **66.0** |

### Real-Time Audio Segmentation (Ternary: SU / Gap / Pause)

| Method | Synthetic F1 | Synthetic IoU | Real F1 | Real IoU |
|--------|-------------|--------------|---------|----------|
| VAP | 92.9 | 87.7 | 17.6 | 10.7 |
| GRU | 85.5 | 76.2 | 24.8 | 14.7 |
| Wav2vec 2.0 | **94.7** | **90.2** | **30.3** | **17.9** |
| **SpeculativeETD** | 94.0 | 88.9 | 28.0 | 16.4 |

SpeculativeETD achieves within 2% IoU of Wav2vec 2.0 on synthetic data while using dramatically less computation.

### Computational Efficiency (FLOPs)

| Method | Compute (MFLOPs) |
|--------|-----------------|
| VAP | 10,354.98 |
| GRU | 45.34 |
| Wav2vec 2.0 | 34,971.68 |
| **SpeculativeETD** | **919.64** (45.34 + 874.30) |

SpeculativeETD uses **38x fewer FLOPs** than Wav2vec 2.0 alone and **11x fewer** than VAP.

### On-Device Latency (iPhone 12 mini)

| Model | Load (ms) | Init (ms) | Execute (ms) |
|-------|-----------|-----------|-------------|
| Wav2vec 2.0 | 874.06 | 17.89 | 1500.32 |
| GRU (SpeculativeETD) | 1.16 | 3.85 | **0.26** |

GRU inference latency: **0.26ms per 100ms interval** -- well within real-time constraints.

## Relevance to Turn-Taking / End-of-Turn Detection

This paper is directly relevant to BabelCast's end-of-turn detection needs:

1. **Architecture template**: The SpeculativeETD two-tier design (lightweight on-device + heavier server-side) maps directly to our Pipecat pipeline. We could use a tiny GRU alongside our existing Silero VAD for continuous speech detection, then invoke a heavier model only at silence boundaries.

2. **ETD Dataset**: First public dataset for this exact task. Can be used directly for fine-tuning our end-of-turn classifier, or as a template for creating domain-specific data.

3. **Three-state formulation (SU/Pause/Gap)**: Clean problem definition that aligns with our needs -- distinguishing mid-utterance pauses from actual turn completions to avoid premature LLM invocations.

4. **Practical latency numbers**: GRU at 0.26ms per inference on mobile hardware confirms feasibility of real-time on-device ETD. Our gateway server has far more compute available.

5. **Real-world gap**: The significant accuracy drop from synthetic to real data (IoU: 88.9 -> 16.4) highlights that ETD on real conversational data remains an open challenge, motivating domain-specific fine-tuning for our meeting translation use case.

6. **Baseline comparisons**: Provides clear benchmarks for VAP, GRU, and Wav2vec models that we can use as reference points for our own models.
