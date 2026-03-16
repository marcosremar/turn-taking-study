---
title: "How Much Does Prosody Help Turn-taking? Investigations using Voice Activity Projection Models"
authors:
  - Erik Ekstedt
  - Gabriel Skantze
year: 2022
source_url: "https://doi.org/10.18653/v1/2023.acl-long.304"
date_converted: 2026-03-16
---

## Abstract

Investigates the role of prosody in turn-taking using the Voice Activity Projection (VAP) model, which incrementally models upcoming speech activity of interlocutors in a self-supervised manner without relying on explicit annotation of turn-taking events or explicit prosodic feature modeling. Through systematic manipulation of the speech signal (F0 flattening, intensity flattening, low-pass filtering, duration averaging, F0 shifting), the authors show that VAP models learn to utilize various prosodic aspects of speech for turn-taking prediction. The study uses both aggregate quantitative metrics on long-form conversations (Switchboard corpus) and controlled utterance-level experiments with synthesized short/long phrase pairs.

## Key Findings Relevant to L2 Turn-Taking

### VAP Model Architecture
- Self-supervised model trained on raw waveforms (no hand-crafted features)
- Predicts future voice activity for both speakers at every time frame
- Tested at 20Hz, 50Hz, and 100Hz frame rates
- Trained on **Switchboard corpus** (English telephone conversations, 260h)
- Evaluated on 4 zero-shot tasks: shift vs. hold, shift prediction at voice activity, backchannel prediction, backchannel vs. turn-shift

### Prosodic Perturbation Results
Five signal manipulations tested:
1. **Low-pass filter** (removes phonetic info, keeps F0 + intensity): Largest overall impact -- phonetic information is crucial
2. **F0 flat** (removes pitch contour): Second most impactful -- intonation is important for disambiguating turn completion
3. **Intensity flat** (removes loudness variation): Comparable impact to F0
4. **F0 shift** (arbitrary pitch shift): Minimal effect on slower models (50Hz), larger effect on faster models (100Hz)
5. **Duration average** (normalizes segment durations): Least important individual cue

### Key Prosodic Finding: F0 Contour for Turn Projection
- At **syntactically ambiguous completion points** (where lexical information is identical for both short and long utterances), the VAP model correctly uses prosody to distinguish turn-yielding from turn-holding
- **Higher F0 rise + longer duration** at the last syllable = turn completion signal
- The model predicts turn shifts **before** the utterance is actually complete -- it projects completions from prosodic dynamics
- Even when F0 is flattened, the model still partially distinguishes hold/shift, indicating **redundant information in intensity and/or duration**

### Relative Importance of Prosodic Cues
1. **Phonetic information** (segmental, captured by spectral content): Most important overall
2. **F0 contour** (intonation): Most important for disambiguating syntactically equivalent completion points
3. **Intensity**: At least as important as pitch for general turn-taking
4. **Duration**: Less important, but contributes as redundant cue

### Model Frame Rate Findings
- 20Hz and 50Hz models: comparable performance, more robust to perturbations
- 100Hz models: more sensitive to phonetic information and acoustic artifacts
- Lower frame rates preferred for computational efficiency and robustness

## Specific Data Points

- Human inter-turn gap: ~200ms average (Levinson & Torreira 2015)
- Switchboard corpus: 2,400 dyadic telephone conversations, 260 hours
- Short/long phrase pairs: 9 pairs, 10 TTS voices each (5 male, 5 female)
- Three evaluation regions at short completion point: hold (start to -200ms), predictive (-200ms to end), reactive (last frame)

## Implications for Turn-Taking Detection in L2 Speech

1. **VAP models work without explicit prosodic features**: The self-supervised approach learns prosodic patterns from raw audio. This is crucial for L2 speech where prosodic patterns deviate from native norms -- the model can potentially learn L2-specific patterns if fine-tuned on L2 data.

2. **F0 contour is the key disambiguation signal**: When words alone cannot determine turn boundaries (common in L2 speech with simpler syntax), the pitch contour becomes the primary cue. French speakers' intonation patterns in Portuguese will be a critical signal for the model.

3. **Prosody is redundant and multi-dimensional**: Even removing one prosodic dimension (F0 or intensity) doesn't completely collapse turn-taking prediction. This redundancy is useful for L2 speech where some prosodic cues may be atypical.

4. **50Hz frame rate is optimal**: For our Pipecat implementation, a 50Hz (20ms frame) VAP model provides the best balance of performance, robustness, and computational efficiency. This aligns with our current 16kHz/512-sample (32ms) Silero VAD window.

5. **Pre-completion projection**: The VAP model predicts turn shifts BEFORE the speaker finishes, based on prosodic dynamics. This is essential for reducing response latency in real-time systems. For L2 speakers, this projection may be less reliable due to non-standard prosodic patterns, requiring L2-specific fine-tuning.

6. **Cross-linguistic transfer needed**: The model was trained on English (Switchboard). For French speakers in Portuguese, it needs fine-tuning on Romance language conversation data. The underlying prosodic principles (F0 rise at completion, intensity patterns) likely transfer across Romance languages, but language-specific patterns must be learned.
