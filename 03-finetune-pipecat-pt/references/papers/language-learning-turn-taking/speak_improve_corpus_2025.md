# Speak & Improve Corpus 2025: L2 English Speech Corpus for Language Assessment and Feedback

**PDF:** `speak_improve_corpus_2025.pdf`
**Source:** https://arxiv.org/pdf/2412.11986
**Authors:** Kate Knill, Diane Nicholls, Mark J.F. Gales, Mengjie Qian, Pawel Stroinski (Cambridge University)
**Venue:** arXiv 2024 (associated with Speak & Improve Challenge 2025)

## Summary

Introduces the Speak & Improve Corpus 2025, the most comprehensive publicly available L2 English learner speech corpus to date. Contains ~315 hours of L2 English learner audio with proficiency scores, plus a 55-hour subset with manual transcriptions (including disfluencies) and grammatical error correction annotations.

## Corpus Details

### Scale
- ~315 hours of L2 English learner speech
- ~55 hours manually transcribed with disfluencies and error labels
- ~950 fully annotated test submissions
- ~2,500+ submissions with proficiency scores only
- Collected from 1.7 million users of the Speak & Improve platform (Dec 2018 - Sep 2024)

### Speaker Diversity
- Speakers from across the globe (wide range of L1 backgrounds)
- CEFR proficiency levels: A2 (Elementary) to C1 (Advanced)
- Majority of data in B1-B2+ range (most common learner levels)

### Test Structure (5 parts, based on Linguaskill Speaking Test)
1. **Interview**: 8 questions about themselves (10-20s each)
2. **Read Aloud**: 8 sentences (not included in corpus)
3. **Long Turn 1**: 1-minute opinion on a topic
4. **Long Turn 2**: 1-minute presentation about a graphic
5. **Communication Activity**: 5 questions on a topic (20s each)

### Three-Phase Annotation
1. **Phase 1 - Scoring**: Audio quality score (3-5) + holistic CEFR score (1-6 per part)
2. **Phase 2 - Transcription**: Manual transcription including disfluencies (hesitations, false starts, repetitions), code-switching, pronunciation errors, phrase boundaries
3. **Phase 3 - Error Annotation**: Grammatical error correction on fluent transcriptions (disfluencies removed first)

### Annotation Tags (Phase 2)
- Word-level: `backchannel`, `disfluency`, `partial`, `pronunciation`
- Word tags: `hesitation`, `code-switch`, `foreign-proper-noun`, `unknown`
- Phrase tags: `speech-unit-incomplete`, `speech-unit-statement`, `speech-unit-question`

### Data Split
| Set | Submissions | Utterances | Hours (Trans) | Hours (SLA) | Words (Trans) |
|-----|-------------|------------|---------------|-------------|---------------|
| Dev | 438 | 5,616 | 22.9 | 35.3 | 140k |
| Eval | 442 | 5,642 | 22.7 | 35.4 | 140k |
| Train | 6,640 | 39,490 | 28.2 | 244.2 | 170k |

## Key Innovations
- First corpus to provide **audio with grammatical error corrections** for spoken GEC research
- Handles complexities unique to spoken GEC: disfluencies, varied accents, spontaneous speech
- Available for non-commercial academic research via ELiT website

## Relevance to Turn-Taking for Language Learners

- **Disfluency patterns by proficiency level**: The corpus captures how hesitations, false starts, and pauses vary across CEFR levels -- directly relevant for training turn-taking models that must distinguish L2 thinking pauses from turn-yielding pauses
- **L1-dependent pronunciation**: Wide L1 diversity means the corpus captures how different L1 backgrounds affect speech patterns, including prosodic cues used in turn-taking
- **Monologic but structured**: While the tasks are monologic (not dialogic), the hesitation and disfluency annotations provide ground truth for understanding L2 speaker timing patterns
- **Phrase boundary annotations**: Can be used to study how L2 speakers signal phrase/turn boundaries differently from L1 speakers
- **Potential training data**: The 315 hours of scored L2 speech could be used to fine-tune ASR systems (like Whisper) that are more robust to L2 speech patterns, which is a prerequisite for accurate turn-taking in L2 conversations
- **Code-switching annotations**: Relevant for multilingual learners who may switch languages mid-turn, confusing standard turn-taking detectors
