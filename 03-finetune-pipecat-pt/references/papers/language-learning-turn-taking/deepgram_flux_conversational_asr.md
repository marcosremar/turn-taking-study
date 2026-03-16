# Introducing Flux: Conversational Speech Recognition

**Source:** https://deepgram.com/learn/introducing-flux-conversational-speech-recognition

## Overview

Deepgram Flux is described as "the first real-time Conversational Speech Recognition model built for voice agents." It addresses the critical challenge of determining when users have finished speaking by integrating turn detection directly into the ASR model.

## Core Problem

Traditional ASR systems were designed for transcription, not real-time conversation. Voice agent developers face an impossible tradeoff:

- **Aggressive approach**: Agents interrupt mid-sentence, destroying user trust
- **Conservative approach**: Robotic pauses damage engagement

The industry currently patches this gap by combining separate systems -- VADs, endpointing layers, and ASR pipelines -- creating complexity and latency issues.

## Key Technical Innovations

### Native Turn Detection

Flux integrates turn-taking into the same model that produces transcripts:

- Semantic awareness (distinguishing "because..." from completed thoughts)
- Fewer false cutoffs through full contextual understanding
- Eliminated pipeline delays since transcripts are ready when turns end

### Performance Benchmarks

- **Median latency reduction**: 200-600ms versus pipeline approaches
- **False interruption reduction**: ~30% fewer compared to alternatives
- **P90/P95 latency**: 1-1.5 seconds for most detections
- **Fast-response rate**: Majority of detections occur within 500ms

### Transcription Quality

- Matches Nova-3 on WER and WRR
- Preserves keyterm prompting capabilities
- Lowest WER on conversational audio benchmarks

### Voice Agent Quality Index (VAQI)

Flux ranked first in overall conversation quality metrics when tested on challenging real-world audio with background noise and disfluencies.

## Developer API

Flux replaces complex multi-system pipelines with two core events:

- **StartOfTurn**: User begins speaking (enables barge-in)
- **EndOfTurn**: High confidence turn completion

Configuration:
- `eot_threshold`: Confidence level (default 0.7)
- `eot_silence_threshold_ms`: Fallback silence duration (default 5000ms)

### Eager End-of-Turn Detection

For latency-critical applications:
- **EagerEndOfTurn**: Medium confidence (150-250ms earlier)
- **TurnResumed**: User continued speaking
- Tradeoff: 50-70% increase in LLM calls for faster responses

## Integration Partners

- Jambonz, Vapi, LiveKit, Pipecat, Cloudflare

## Future Roadmap

- Self-hosted deployment support
- Multilingual support
- Word-level timestamps
- Selective listening and backchanneling identification
