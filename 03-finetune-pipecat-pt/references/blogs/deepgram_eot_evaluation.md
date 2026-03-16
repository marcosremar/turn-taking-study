---
title: "Evaluating End-of-Turn Detection Models"
source: https://deepgram.com/learn/evaluating-end-of-turn-detection-models
date_accessed: 2026-03-16
---

# Evaluating End-of-Turn (Turn Detection) Models

## Overview

Natural voice agent interactions depend critically on accurate turn-taking behavior. End-of-Turn (EoT) detection -- predicting when a speaker has finished and awaits a response -- is essential for creating conversational agents that feel responsive rather than interruptive.

Deepgram developed a comprehensive evaluation methodology for turn detection after finding existing approaches inadequate. Their approach prioritizes real-world performance over proxy metrics, evaluating "complete conversations between humans" rather than isolated turns.

## Full Conversational Evaluation

Rather than testing individual turns, Deepgram analyzes entire conversations. This methodology reflects realistic scenarios where:

- **Natural labeling**: Conversation partners provide high-quality, low-latency ground truth detection
- **Variable detection budgets**: Different turns demand faster responses based on context (simple queries versus complex reasoning)
- **Natural pause patterns**: Pre-turn silences enable evaluation of start-of-speech detection and backchannel phenomena

The team labeled over 100 hours of real conversations with ground truth transcripts and timestamped EoT labels. Notably, refining their annotation specification from "End-of-Thought" to "End-of-Turn" improved annotator confidence from 5/10 to 8.5/10 by reducing ambiguity.

## The Challenge with Timestamps

Human annotations revealed an unexpected problem: annotators consistently left small gaps between speech completion and label placement. Deepgram discovered their system frequently detected EoT within these gaps -- appearing as false positives under naive temporal alignment.

### Forced Alignment Solution

Rather than trusting raw human timestamps, Deepgram applied "forced alignment to update the human timestamps," extracting more precise end-of-speech markers. While conservative human placement might reflect natural pauses before responses, voice agents benefit from fastest possible detection since "their turn starts are frequently delayed relative to EoT detection due to extra latency associated with LLM and TTS generation."

## Sequence Alignment for Improved Evaluation

Standard temporal matching proved insufficiently precise. Deepgram adopted sequence alignment -- treating turn boundaries as special tokens (`[EoT]`) within transcripts, similar to Word Error Rate (WER) calculation.

**Impact**: This shift yielded "3-5% absolute increases in precision and recall across models," with "manual investigation of results suggested the new values were more representative of the true performance."

The approach handled all detector types: all-in-one STT solutions, text-based detectors, and audio-only models (using Nova-3 for inter-turn transcription).

## Handling Dropped Turns

A modified Levenshtein algorithm addressed cases where intermediate turns were missed. Beyond simple edit distance, the system "determines the best alignment not only based on overall edit distance but also based on the most likely alignment between EoT tokens," preventing misalignment when turns are dropped.

## Start-of-Turn (SoT) Evaluation

Complementing EoT analysis, Deepgram evaluated "start-of-turn (SoT)" detection -- identifying when users resume speaking, critical for handling interruptions or "barge-in" scenarios.

**Flux performance metrics**:
- Detection within ~100-200ms of first word onset
- False positive rate: less than or equal to 1-2%
- Analysis used word start times as detection targets (representing unachievable lower bounds)

Negative latency directly indicates false positives -- detecting speech before it actually begins.

## Future Directions

The team highlighted emerging evaluation frontiers: "one could combine the various aspects of turn detection, such as false positive rate and latency, into single quality metrics." They specifically mentioned their Voice Agent Quality Index (VAQI) framework and anticipated incorporating semantic/conversational flow considerations -- enabling "stratified or weighted metrics that reflect that some turns merit faster responses than others."
