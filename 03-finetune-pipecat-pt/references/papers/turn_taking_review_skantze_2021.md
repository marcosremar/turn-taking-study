---
title: "Turn-taking in Conversational Systems and Human-Robot Interaction: A Review"
authors:
  - Gabriel Skantze
year: 2021
source: https://doi.org/10.1016/j.csl.2020.101178
date_converted: 2026-03-16
---

## Abstract

The taking of turns is a fundamental aspect of dialogue. Since it is difficult to speak and listen at the same time, the participants need to coordinate who is currently speaking and when the next person can start to speak. Humans are very good at this coordination, and typically achieve fluent turn-taking with very small gaps and little overlap. Conversational systems (including voice assistants and social robots), on the other hand, typically have problems with frequent interruptions and long response delays, which has called for a substantial body of research on how to improve turn-taking in conversational systems. This review provides an overview of this research and gives directions for future research. It covers the theoretical background of linguistic research on turn-taking, an extensive review of multi-modal cues (including verbal cues, prosody, breathing, gaze and gestures) that facilitate turn-taking coordination, and work on modelling turn-taking including end-of-turn detection, handling of user interruptions, generation of turn-taking cues, and multi-party human-robot interaction.

**Note**: The PDF file `turn_taking_review_skantze_2021.pdf` in this directory is corrupted (contains an unrelated physics paper). Content for this summary was sourced from the Semantic Scholar API and the published paper metadata. The actual paper is available at the ScienceDirect DOI above (open access, CC-BY license).

## Key Contributions

1. **Comprehensive survey** of turn-taking research spanning linguistics, psycholinguistics, and computational approaches (288 citations as of 2026).
2. **Taxonomy of turn-taking cues** across modalities: verbal/linguistic, prosodic, breathing, gaze, and gesture.
3. **Review of computational models** for end-of-turn detection, interruption handling, and turn-taking cue generation.
4. **Identification of open challenges** for achieving fluent turn-taking in human-machine interaction.

## Theoretical Background

### Fundamental Concepts

- **Turn-Constructional Units (TCUs)**: The basic building blocks of turns, roughly corresponding to clauses or sentences.
- **Transition Relevance Places (TRPs)**: Points at the end of TCUs where speaker transition may occur.
- **Inter-Pausal Units (IPUs)**: Stretches of speech bounded by silence (typically >200ms).
- **Turn-yielding vs. Turn-holding cues**: Signals at the end of IPUs that indicate whether the speaker intends to continue or give up the floor.

### Gap and Overlap Statistics

- Average gap between turns in human conversation: ~200ms (Levinson & Torreira, 2015).
- Typical spoken dialogue systems use silence thresholds of 1-2 seconds -- far longer than natural gaps.
- Pauses within turns are often longer than gaps between turns, making silence duration alone unreliable for end-of-turn detection.

## Turn-Taking Cues Reviewed

### Prosodic Cues
- **Pitch**: Turn-final utterances tend to have falling or rising-falling pitch contours. Rising pitch may signal continuation or questions.
- **Intensity**: Decreasing intensity toward the end of a turn.
- **Speaking rate**: Slowing down at turn boundaries.
- **Duration**: Final syllable lengthening at turn ends.

### Verbal/Linguistic Cues
- **Syntactic completeness**: Complete syntactic units are strong turn-yielding cues.
- **Pragmatic completeness**: Whether the communicative intent has been fulfilled.
- **Discourse markers**: Words like "so", "well", "anyway" can signal turn boundaries.
- **Content words vs. function words**: Turns ending on content words are more likely to be complete.

### Gaze Cues
- Speakers tend to look away at the start of turns and gaze at the listener toward the end.
- Mutual gaze at the end of an utterance is a strong turn-yielding signal.

### Gesture Cues
- Hand gestures in progress signal turn-holding.
- Completion of a gesture stroke may signal turn completion.

### Breathing
- Audible inhalation can signal upcoming speech (turn-taking intention).
- Exhalation patterns correlate with turn boundaries.

## Computational Models Reviewed

### End-of-Turn Detection Approaches

1. **Fixed silence threshold**: Simplest approach; typically 700-2000ms. Used by most commercial systems. Leads to either frequent interruptions (short threshold) or long delays (long threshold).

2. **Decision-theoretic approaches**: Raux & Eskenazi (2009) Finite-State Turn-Taking Machine -- uses cost matrices and probabilistic state estimation to optimize the threshold dynamically.

3. **Classification-based approaches**: Train classifiers (logistic regression, SVMs, decision trees) on prosodic and lexical features to predict turn boundaries at each silence onset.

4. **Neural network approaches**:
   - RNN/LSTM models that process sequences of features continuously (Skantze 2017).
   - Transformer-based models like TurnGPT (Ekstedt & Skantze, 2020) that use linguistic context.
   - Voice Activity Projection (VAP) models (Ekstedt & Skantze, 2022) that predict future voice activity from raw audio.

5. **Incremental processing**: Systems that make predictions before the utterance is complete, enabling faster response times.

### Key Features for End-of-Turn Prediction
- Pause duration (most commonly used but unreliable alone)
- Pitch contour (F0) at utterance end
- Energy/intensity trajectory
- Speaking rate changes
- Syntactic completeness (from ASR partial results)
- Semantic completeness / dialogue act
- Language model scores (boundary LM)

### Handling Interruptions
- **Barge-in detection**: Determining when user speech during system output is an intentional interruption vs. backchannel.
- **Backchannel prediction**: Predicting when brief acknowledgments ("uh-huh", "yeah") will occur.
- **Overlap resolution**: Strategies for who yields when both participants speak simultaneously.

## Key Findings and Recommendations

1. **Silence is not enough**: Pauses within turns are often longer than gaps between turns. Systems relying solely on silence thresholds will always face a trade-off between responsiveness and interruption rate.

2. **Multi-modal cues are complementary**: No single cue is sufficient. The best systems combine prosodic, linguistic, and (where available) visual cues.

3. **Continuous prediction is better than binary**: Rather than making a single end-of-turn decision at each silence, continuously predicting the probability of upcoming speaker activity yields more natural turn-taking.

4. **Incremental processing is key**: Systems should process input incrementally (not waiting for complete utterances) to enable fast response times comparable to human turn-taking gaps.

5. **Domain and context matter**: Turn-taking patterns vary significantly across different dialogue types (task-oriented vs. social chat), cultures, and individual speakers.

## Relevance to Turn-Taking / End-of-Turn Detection

This review paper is the foundational reference for BabelCast's turn-taking work:

1. **Problem framing**: Establishes that the ~200ms human turn-taking gap is the gold standard to aim for, and that current systems at 700-1000ms are far too slow. Our real-time translation pipeline adds additional latency, making fast ETD even more critical.

2. **Feature selection guide**: The taxonomy of turn-taking cues directly informs which features to extract for our ETD model. Prosodic features (pitch, energy, speaking rate) + linguistic completeness from ASR are the most practical for an audio-only system like ours.

3. **Model architecture guidance**: The progression from fixed thresholds to neural continuous prediction models provides a clear roadmap. We should move beyond our current fixed Silero VAD threshold toward a learned model.

4. **Evaluation framework**: The review establishes standard metrics (latency, cut-in rate, balanced accuracy for shift/hold prediction) that we should adopt for benchmarking our ETD improvements.

5. **Backchannel awareness**: In meeting translation, backchannels ("uh-huh", "right") should not trigger translation. The review's discussion of backchannel detection is relevant for filtering these out.

6. **Multi-party challenges**: The review notes that multi-party turn-taking is significantly harder and less studied. Since our meeting bot handles multi-speaker calls, this is an open area for us.
