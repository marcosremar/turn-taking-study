---
title: "How to Build Smarter Turn Detection for Voice AI"
source: https://blog.speechmatics.com/semantic-turn-detection
date_accessed: 2026-03-16
---

# How to Build Smarter Turn Detection for Voice AI

**Published:** May 12, 2025 | **Author:** Aaron Ng, Machine Learning Engineer | **Read Time:** 13 minutes

## Introduction

Voice AI systems face a fundamental challenge: determining when users have finished speaking. Traditional approaches relying on fixed silence periods (like 500ms) create frustrating interruptions. This article explores how semantic understanding can improve turn detection.

## The Problem with Traditional Voice Activity Detection

VAD systems detect speech boundaries through audio patterns alone. As the article explains, "VAD only understands audio patterns. It knows _when_ there's speech and when there isn't. What it doesn't know is _why_ there's a pause."

The example demonstrates this limitation: when a customer pauses to check information ("Sure it's 123 764... (pauses to check their notes)"), VAD incorrectly interprets the silence as turn completion and triggers an interruption.

## Semantic Turn Detection Solution

Rather than relying solely on silence detection, the proposed approach uses instruction-tuned Small Language Models (SLMs) to understand conversational context. These models contain fewer than 10 billion parameters, enabling fast local inference critical for voice interactions.

### Key Benefits

- **Reduced Latency**: Local SLM inference avoids the 500ms+ delays of external API calls
- **Cost Savings**: Fewer false interruptions mean fewer unnecessary LLM API calls for responses that get discarded
- **Better UX**: Systems recognize contextual pauses versus actual turn completion

## Technical Implementation

### Core Mechanism

The approach monitors the probability of the `<|im_end|>` token -- a special ChatML marker indicating turn completion. When this token's probability is high, the model predicts the user has finished speaking.

The article illustrates this with examples:
- "Can I have two chicken McNuggets and" -> Low `<|im_end|>` probability (incomplete thought)
- "I have a problem with my card" -> Higher `<|im_end|>` probability (complete thought)

### Architecture Components

**1. Message Tokenization**

Messages are formatted using ChatML structure with special tokens (`<|im_start|>`, `<|im_end|>`, `<|im_sep|>`). The implementation removes the final `<|im_end|>` token since the model predicts its presence.

**2. Model Inference**

The code uses `AutoModelForCausalLM` to extract logits for the final token position, computing log-softmax probabilities across the vocabulary.

**3. Token Probability Extraction**

The system identifies the `<|im_end|>` token among top-k predictions (k=20) and converts log probabilities to standard probabilities using exponential transformation.

### Code Example Structure

The implementation includes an `EndOfTurnModel` class with:

- `_convert_messages_to_chatml()`: Formats conversations in ChatML format
- `get_next_token_logprobs()`: Performs local inference to retrieve next-token probabilities
- `process_result()`: Extracts target token probabilities
- `predict_eot_prob()`: Orchestrates the full pipeline returning probability scores

**Model Details**: The implementation uses `SmolLM2-360M-Instruct` from Hugging Face, chosen for efficiency and CPU compatibility.

## Configuration Parameters

- **MAX_HISTORY**: 4 messages (recent context window)
- **DEFAULT_THRESHOLD**: 0.03 (default probability threshold)
- **Top-k consideration**: 20 tokens

## Practical Considerations

### Token Selection Strategy

Beyond `<|im_end|>`, punctuation marks (periods, question marks, exclamation points) can signal thought completion. Hybrid approaches monitoring multiple tokens may improve accuracy but risk introducing noise.

### Hybrid Approach with VAD

The article recommends combining semantic detection with VAD:
- VAD provides high recall detecting speech presence
- Semantic turn detection improves precision to reduce false interruptions
- Dynamic grace periods can adjust based on speaking patterns

### Threshold Determination

The default 0.03 threshold serves as a starting point. "The optimal threshold depends on your specific SLM and can vary significantly between models." Precision-focused optimization using representative test sets is recommended.

### Multilingual Support

Supporting multiple languages requires instruction-tuned SLMs trained on diverse multilingual data, accounting for varied grammar and conversational norms.

### Fine-Tuning Opportunities

While general SLMs provide solid conversational understanding, task-specific fine-tuning on domain conversational data can improve accuracy for specialized terminology or industry-specific use cases.

## Future Directions

The article acknowledges limitations: "True conversational understanding requires more than just reading text or listening for gaps -- it needs an audio-native model" that processes tone, cadence, hesitation, and emphasis directly from audio signals.

## Resource Reference

A complete implementation is available on GitHub at the repository referenced in the article.
