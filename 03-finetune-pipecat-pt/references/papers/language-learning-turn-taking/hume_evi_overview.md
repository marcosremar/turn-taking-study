# Hume EVI (Empathic Voice Interface) Overview

**Source:** https://dev.hume.ai/docs/speech-to-speech-evi/overview

## Overview

Hume's Empathic Voice Interface (EVI) is a real-time voice AI system that analyzes emotional nuances in speech. It processes "tune, rhythm, and timbre of speech" to enable more natural interactions between humans and AI systems.

## Core Capabilities

- **Transcription**: Rapid ASR with expression measurements aligned to each sentence
- **Language Generation**: Speech-language model, optionally integrated with Anthropic and OpenAI APIs
- **Voice Response**: Streamed speech generation via the speech-language model
- **Low Latency**: Immediate responses by running integrated models on unified infrastructure

## Empathic AI Features

- **Timing Detection**: Determines appropriate response moments using vocal tone analysis
- **Prosody Understanding**: Measures user speech characteristics via integrated prosody modeling
- **Tonal Matching**: Generates responses reflecting the user's emotional state (apologetic for frustration, sympathetic for sadness)
- **Expression-Aware Responses**: Crafts linguistically appropriate answers based on vocal expression
- **Interruptibility**: Stops speaking when interrupted and resumes with proper context
- **Multilingual Support**: EVI 4-mini handles 11 languages; EVI 3 supports English only

## Version Comparison

| Feature | EVI 3 | EVI 4-mini |
|---------|-------|------------|
| Quick responses | Yes | No |
| Supplemental LLM required | Optional | Required |
| Languages | English only | 11 languages |

## API Architecture

- WebSocket connections for real-time dialogue
- Authentication via API keys or access tokens (query parameters)
- Session concurrency limits by subscription tier
- Maximum 30-minute session duration
- 16 MB message size limit

## Developer Resources

- Quickstart guides for Next.js, TypeScript, and Python
- WebSocket and REST API references
- Sample code repositories

## Relevance to Turn-Taking

EVI's approach to turn-taking is notable because it uses prosody and emotional analysis rather than just silence detection. The system analyzes vocal tone to determine when a user has finished speaking, which is particularly relevant for language learners who may have longer pauses, hesitations, and non-standard prosodic patterns.
