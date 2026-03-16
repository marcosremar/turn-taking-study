# Evaluating End-of-Turn Detection Models

**Source:** https://deepgram.com/learn/evaluating-end-of-turn-detection-models
**Authors:** Jack Kearney (Staff Research Scientist) & Chau Luu (Senior Research Scientist)
**Published:** October 28, 2025

## Overview

Natural voice agent interactions depend critically on accurate turn-taking behavior. An agent responding too quickly may interrupt users, while one that's overly cautious feels sluggish and unresponsive.

Detecting End-of-Turn (EoT) -- recognizing when a speaker has finished and awaits a response -- is fundamental to building conversational voice systems. Multiple solutions exist, ranging from integrated ASR+EoT systems like Deepgram Flux and AssemblyAI Universal-Streaming, to audio-focused approaches like Pipecat Smart Turn and Krisp Turn-Taking, and transcript-based solutions such as LiveKit's EoU model.

The Deepgram team needed rigorous evaluation methodology to ensure Flux delivered superior EoT detection. Rather than relying on proxy metrics or isolated turn analysis, they developed a novel evaluation framework centered on complete, real conversational data and sophisticated sequence alignment techniques.

## Full Conversational Evaluation

The team evaluated models against entire human conversations rather than individual turns. This approach captures realistic conversational dynamics that isolated turn analysis misses:

- **High-quality labels:** Natural counterparty speech provides reliable turn boundary detection
- **Realistic detection budgets:** Not all turns allow equal response latency; simple queries demand faster responses than complex ones
- **Natural pause patterns:** Pre-turn silence can be studied alongside turn endings, enabling investigation of start-of-speech detection and potential backchanneling

The research dataset comprised over 100 hours of genuine conversations with groundtruth transcripts and timestamped EoT annotations. An important lesson emerged during labeling: using "End-of-Thought" instead of "End-of-Turn" created ambiguity. The revised specification -- asking annotators to mark where a voice agent should naturally begin speaking -- improved self-reported confidence from 5/10 to 8.5/10.

## The Trouble with Timestamps

Initial attempts to match predicted EoT times against human-annotated labels revealed a critical problem: human annotators tended to be somewhat conservative in label placement, leaving a small gap between actual speech cessation and marked timing. This meant Flux sometimes detected EoT within these gaps, yet would register as incorrect using purely temporal comparison.

To address this, researchers employed forced alignment to refine human timestamps, anchoring labels to precise speech endings extracted through acoustic analysis. While critics might argue that human labels better represent when naturally to speak (with a small pause), voice agent contexts differ from human conversation -- agents have separate latency from LLM and TTS processing, making early detection advantageous.

### The Limitation of Time-Based Alignment

A naive approach would simply match each groundtruth EoT to the earliest detection falling between turn end and the next speaker's start. However, this imposes unrealistic constraints: it forbids detection before acoustic speech completion, ignoring that sophisticated models might predict EoT before a word finishes.

Relaxing this requirement by allowing slightly early detection introduces new hyperparameters without substantially improving reliability, according to manual review.

## Using Sequence Alignment to Improve Turn Boundary Detection Evaluation

The breakthrough came from applying sequence alignment -- the same technique used to calculate word error rate (WER) in speech-to-text -- to turn detection evaluation. By treating turn boundaries as special tokens (`[EoT]`) in transcripts, researchers leveraged transcript context to improve EoT prediction-to-groundtruth matching.

### Example of Sequence Alignment Advantage

Consider this scenario:

```
TRUTH:      Hi  Chau! [EoT-1] I'm fine thanks. [EoT-2] Sure that sounds great. [EoT-3]
PREDICTION: Hi!                                 [EoT]  Sure that sounds great.  [EoT]
```

Purely temporal alignment would penalize the first `[EoT]` prediction with both a false positive and false negative because it appeared 180ms early. However, sequence alignment recognizes the prediction aligns reasonably well with the groundtruth structure -- just slightly ahead of time.

### Performance Impact

Switching from pure temporal to sequence-alignment-based evaluation yielded dramatic improvements: 3-5% absolute increases in precision and recall across models for all-in-one solutions, text-based detectors, and audio-only systems. Manual investigation confirmed these improvements accurately reflected true performance.

## Handling Dropped Turns

Standard Levenshtein alignment can produce ambiguous results when turns are dropped. The team modified the Levenshtein algorithm to prioritize `[EoT]`-to-`[EoT]` alignment quality alongside overall edit distance, with timestamp context providing additional evidence.

## Turn Start Evaluation

Beyond EoT, detecting when users begin speaking (Start-of-Turn/SoT) matters for handling interruptions and barge-in scenarios. Faster SoT detection reduces overlapping speech and preserves natural interaction flow.

### Flux SoT Performance

Flux demonstrates robust SoT detection, generally identifying turn starts within approximately 100-200ms of typical first-word duration, with false positive rates below 1-2%.

## Key Takeaways

1. **Complete conversation evaluation** beats isolated turn analysis for capturing realistic dynamics
2. **Sequence alignment** significantly outperforms pure timestamp matching for evaluation accuracy
3. **Forced alignment refinement** of human annotations improves label precision without removing human judgment
4. **Modified Levenshtein algorithms** better handle edge cases like dropped turns
5. **Comprehensive metrics** combining accuracy, latency, and false positives provide fuller performance pictures than isolated measurements
