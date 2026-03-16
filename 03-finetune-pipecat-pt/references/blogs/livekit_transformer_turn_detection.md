---
title: "Using a Transformer to Improve End of Turn Detection"
source: https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection
date_accessed: 2026-03-16
---

# Using a Transformer to Improve End of Turn Detection

**Published:** December 20, 2024 | **Author:** Russ D'Sa | **Reading Time:** 6 minutes

## Overview

End-of-turn detection represents one of the most challenging problems in voice AI applications. The core task involves determining when a user has finished speaking, allowing an AI model to respond without unintentionally interrupting the user.

## Current Approach: Phrase Endpointing

The predominant technique for turn detection is phrase endpointing, typically implemented through voice activity detection (VAD). This method operates as follows:

- Audio packets are processed through a neural network to determine if human speech is present
- If speech is detected, the user continues speaking
- If silence is detected, a timer begins tracking the duration of the absence of detectable human speech
- Once a configured silence threshold passes, an end-of-turn event triggers, allowing LLM inference to begin

LiveKit Agents uses "Silero VAD to detect voice activity and provide timing parameters" to adjust sensitivity. The framework introduces a configurable delay via the `min_endpointing_delay` parameter (default: 500ms) between when VAD detects speech cessation and when LLM inference begins.

## The VAD Limitation

VAD only detects *when* someone is speaking based on audio signal presence. Humans, however, use semantic understanding -- analyzing what is said and how it's expressed -- to determine turn-taking. For example, "I understand your point, but..." would trigger VAD's end-of-turn signal despite the human listener recognizing the speaker intends to continue.

## The EOU (End of Utterance) Model

LiveKit released "an open source transformer model that uses the content of speech to predict when a user has" finished speaking. The model incorporates semantic understanding into turn detection.

### Model Architecture

- **Base Model:** 135M parameter transformer based on SmolLM v2 from HuggingFace
- **Design Choice:** Small model size enables CPU-based, real-time inference
- **Context Window:** Sliding window of the last four conversational turns
- **Language Support:** Currently English transcriptions only; additional languages planned

### How It Works

During user speech, transcriptions from a speech-to-text (STT) service are appended word-by-word to the model's context window. For each final STT transcription, the model generates a confidence-level prediction regarding whether the current context represents turn completion.

The model integrates with VAD by dynamically adjusting the silence timeout: longer silence periods are permitted when EOU indicates the user hasn't finished speaking, reducing interruptions while maintaining responsiveness.

## Performance Results

Compared to VAD alone, the EOU + VAD approach achieves:

- **85% reduction** in unintentional interruptions
- **3% false negative rate** (incorrectly indicating turn continuation)

### Use Cases

The model proves particularly valuable for conversational AI and customer support applications requiring data collection:

- Conducting interviews
- Collecting addresses for shipments
- Gathering phone numbers
- Processing payment information
- Ordering transactions (pizza ordering demonstrated)

## Implementation

The turn detector is packaged as a LiveKit Agents plugin, enabling simple integration through a single additional parameter in the `VoicePipelineAgent` constructor:

```
turn_detector=TurnDetector.from_plugin()
```

Example implementation code is available in the LiveKit agents repository.

## Future Development

Planned improvements include:

- **Multi-language Support:** Extending EOU beyond English
- **Inference Optimization:** Reducing current ~50ms inference latency
- **Context Window Expansion:** Increasing the four-turn window
- **Audio-Based Detection:** Developing models for multimodal systems that process audio directly, accounting for prosodic features like intonation and cadence

The team acknowledges that EOU's text-based training limits its use with natively multimodal models such as OpenAI's Realtime API. An audio-native model is under development to address this constraint.

## Broader Implications

The researchers emphasize that "even humans don't get this right all the time" and that non-verbal cues improve human turn-taking performance. They consider this research essential for developing natural, humanlike AI interactions and anticipate significant community innovation in this area.
