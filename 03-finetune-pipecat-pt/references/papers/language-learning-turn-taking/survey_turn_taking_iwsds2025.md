# Survey of Recent Advances on Turn-taking Modeling in Spoken Dialogue Systems (IWSDS 2025)

**PDF:** `iwsds2025_survey_turn_taking.pdf`
**Source:** https://aclanthology.org/2025.iwsds-1.27.pdf
**Authors:** Galo Castillo-Lopez, Gael de Chalendar, Nasredine Semmar (Universite Paris-Saclay, CEA, List)
**Venue:** IWSDS 2025 (15th International Workshop on Spoken Dialogue Systems Technology)

## Summary

Comprehensive survey of recent methods on turn-taking modeling in spoken dialogue systems, with special attention on studies published after 2021 (building on Skantze 2021's earlier survey). The paper reviews end-of-turn prediction, backchannel prediction, and multi-party conversation turn-taking.

## Key Findings

### The Problem
- Human-human conversation transitions take ~200ms on average
- Current spoken dialogue agents initiate turns after 700-1000ms gaps, resulting in unnatural conversations
- 72% of reviewed works do NOT compare their methods with previous efforts -- a major gap
- Lack of well-established benchmarks to monitor progress

### Three Categories of End-of-Turn Methods
1. **Silence-based**: Simple VAD + silence threshold (e.g., 700ms). Poor user experience.
2. **IPU-based (Inter-Pausal Unit)**: Predictions made after each detected silence. Assume turns cannot be taken while user speaks.
3. **Continuous**: Constantly evaluate end-of-turn regardless of silences (e.g., every 50ms). Most promising.

### Continuous Methods -- State of the Art
- **Voice Activity Projection (VAP)**: Predicts future voice activity for both speakers in a dialogue using cross-attention Transformers on raw audio. Emerging as the dominant approach.
- **TurnGPT**: GPT-2-based model fine-tuned on dialogue datasets for turn-completion prediction based on text features only. Outperforms previous work due to strong context representation.
- Combined prosodic + linguistic features consistently outperform individual feature types (additive effect of turn-taking cues).
- LLMs are currently **inefficient** at detecting mid-utterance turn-taking opportunities (Umair et al., 2024).

### IPU-based Methods
- LSTM-based architectures for prosodic/phonetic/lexical features
- CNN models effective when incorporating visual cues (eye, mouth, head motion)
- Speech acts as auxiliary tasks improve turn-taking performance
- Recent work: instruction fine-tuning on LLMs with HuBERT audio features

### Datasets Used
- **Switchboard** (260h, 2.4K dyadic telephone dialogues) -- most used
- **Fisher Corpus** (1960h, 11.7K topic-oriented telephone conversations)
- **AMI/ICSI Meeting Corpus** (100h/72h, multi-party)
- Datasets predominantly in English; some in Japanese, Mandarin, French, German
- Notable gap: very few datasets for non-English and multi-party scenarios

### Multi-party Conversations
- Much less explored than dyadic scenarios
- Additional complexity: addressee recognition, floor management
- Mainly studied in human-robot interaction contexts
- CEJC (Japanese) and AMI/ICSI (English) corpora used

### Key Challenges Identified
1. No standardized benchmarks for comparing turn-taking models
2. Most research focuses on English; multilingual models needed
3. Integration of visual/multimodal features underexplored in VAP models
4. Real-time deployment challenges
5. Handling interruptions and overlaps poorly addressed

## Relevance to Turn-Taking for Language Learners

- L2 speakers produce longer pauses, more hesitations, and non-standard prosody -- all confuse standard EoT detectors
- The VAP model's continuous prediction approach could be adapted to handle L2 speaker patterns by training on L2 speech data
- TurnGPT's text-based approach might be less affected by L2 pronunciation issues but more affected by grammatical errors
- Multi-party turn-taking (poorly studied) is exactly the scenario in classroom/group language learning
- The finding that combined cues work better than individual ones suggests L2-adapted systems should use both audio and text features
