---
title: "A Survey of Recent Advances on Turn-taking Modeling in Spoken Dialogue Systems"
authors:
  - Galo Castillo-Lopez
  - Gael de Chalendar
  - Nasredine Semmar
year: 2025
source_url: "https://aclanthology.org/2025.iwsds-1.24/"
date_converted: 2026-03-16
---

## Abstract

A comprehensive review of recent methods on turn-taking modeling in spoken dialogue systems. Covers end-of-turn prediction, backchannel prediction, and multi-party turn-taking. Notes that 72% of reviewed works do not compare their methods with previous efforts, and argues that the lack of well-established benchmarks is a key challenge. Provides the first detailed review of datasets used in the field, discusses overlooked limitations, and examines new ideas and approaches since 2021. Published at IWSDS 2025.

## Key Findings Relevant to L2 Turn-Taking

### Turn-Taking Modeling Taxonomy
Three main approaches to end-of-turn (EOU) prediction:
1. **Silence-based**: Uses VAD + silence threshold (e.g., 700ms). Results in poor user experience due to unnaturalness. Current spoken dialogue systems wait 700-1000ms (vs. human 200ms average).
2. **IPU-based (Inter-Pausal Unit)**: Predictions after each detected silence. Assumes turns cannot be taken while user speaks.
3. **Continuous**: Predictions at every frame (e.g., every 50ms) regardless of silence. Most recent and promising approach.

### Voice Activity Projection (VAP) Models -- Current State of the Art
- VAP models predict future voice activity of both interlocutors incrementally
- Self-supervised learning -- no explicit turn-taking event annotation needed
- Best results in Japanese when trained on English + fine-tuned with Japanese data (vs. direct Japanese training)
- Cross-lingual performance is poor without proper label alignment across datasets
- Emerging trend: VAP models dominate continuous methods

### Feature Importance for Turn-Taking
- **Combined features outperform individual ones**: prosody + words > prosody alone or words alone
- **Turn-taking cues have additive effect** in human communication
- Word embeddings + acoustic features together improve backchannel detection
- Gaze, head pose, and non-verbal features enhance predictions when available
- **ASR-based linguistic features**: Fine-tuning wav2vec 2.0 for ASR outperforms acoustic-only features

### Key Datasets
| Dataset | Language | Duration | Dialogues |
|---|---|---|---|
| Switchboard | English | 260h | 2,400 |
| Fisher Corpus | English | 1,960h | 11,700 |
| NoXi Database | en/es/fr/de/it/ar/id | 25h | 84 |
| AMI Meeting Corpus | English | 100h | 175 (multi-party) |

- **Switchboard** is the dominant benchmark: used in 69% of backchannel and 41% of EOU papers
- Most research is on **English and Japanese** -- very limited work on other languages
- IPU silence thresholds vary from **50ms to 1s** across studies -- no standard

### Backchannel Prediction
- Backchannels are short feedback tokens ("mhm", "yeah") produced during another's speech
- Key distinction: backchannel vs. actual interruption vs. non-lexical noise
- Multi-task learning (sentiment + dialogue acts + backchannel) improves prediction
- Context-aware models using BERT text embeddings + wav2vec acoustic embeddings show strong results

### Open Challenges Identified
1. **No standardized benchmarks**: Only 28% of papers compare with prior work
2. **Multilinguality**: Very limited -- most work is English/Japanese only
3. **Multi-party conversations**: Understudied, requires visual channel
4. **Special populations**: Senior adults, mental health disorders need adapted models
5. **LLMs are inefficient** at detecting mid-utterance turn opportunities

## Implications for Turn-Taking Detection in L2 Speech

1. **Continuous VAP models are the right approach**: For our fine-tuning task, continuous frame-level prediction (not silence-threshold-based) is the current state of the art. This aligns with our Pipecat architecture.

2. **Cross-lingual fine-tuning works**: VAP models trained on English and fine-tuned on target language data outperform direct target-language training. This suggests training on English Switchboard + fine-tuning on French-Portuguese conversation data is a viable strategy.

3. **Multi-feature approach is essential**: For L2 speakers, combining prosodic + lexical + timing features provides the best predictions. L2 speakers may have atypical prosody but more predictable syntax, or vice versa. The additive effect of features provides robustness.

4. **Silence threshold tuning**: The survey notes thresholds from 50ms to 1s. For L2 speakers with longer pauses, the threshold must be adjusted upward. Our current 300ms SILENCE_HANGOVER_MS may need extension for L2 French speakers in Portuguese.

5. **French is available in NoXi**: The NoXi database includes French conversations (among 7 languages) with 25h total. This could be a valuable fine-tuning resource for our French-specific model.

6. **Backchannel handling**: L2 speakers may produce non-standard backchannels or transfer French backchannels ("ouais", "mmh") into Portuguese conversation. The model must distinguish these from turn-taking attempts.

7. **Benchmarking gap**: Since 72% of papers don't compare methods, we should establish clear metrics for our L2 turn-taking task from the start, enabling future comparison.
