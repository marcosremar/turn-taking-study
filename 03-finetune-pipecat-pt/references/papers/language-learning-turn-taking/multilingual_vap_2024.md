# Multilingual Turn-taking Prediction Using Voice Activity Projection

**PDF:** `multilingual_vap_2024.pdf`
**Source:** https://arxiv.org/pdf/2403.06487
**Authors:** Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze
**Venue:** LREC-COLING 2024 (Kyoto University + KTH Royal Institute of Technology)

## Summary

Investigates whether the Voice Activity Projection (VAP) model -- a continuous turn-taking predictor that works on raw audio -- can be applied across languages. Tests on English, Mandarin, and Japanese (three different language families: Germanic, Sino-Tibetan, Japonic).

## VAP Model Architecture

- **Input**: Stereo audio (one channel per speaker), up to 20 seconds, 16kHz, 50Hz frame rate
- **Audio Encoder**: Contrastive Predictive Coding (CPC) pre-trained on English LibriSpeech, frozen during training
- **Architecture**: CPC encoder -> Self-attention Transformer (per channel) -> Cross-attention Transformer (interaction between channels) -> Linear layers for VAP + VAD predictions
- **Output**: Predicts joint voice activity of both speakers over a 2-second future window, divided into 4 bins (0-200ms, 200-600ms, 600-1200ms, 1200-2000ms), yielding 256 possible states

## Research Questions & Answers

### RQ1: Can a monolingual model transfer to other languages?
**No, not well.** A VAP model trained on one language does not make good predictions on other languages. Cross-lingual performance drops significantly.

### RQ2: Can a single multilingual model match monolingual performance?
**Yes.** A multilingual model trained on all three languages performs on par with monolingual models across all languages. This is a key finding -- one model can handle multiple languages.

### RQ3: Has the multilingual model learned to identify language?
**Yes.** Analysis shows the multilingual model has learned to discern the language of the input signal, suggesting it develops language-specific internal representations.

### RQ4: How important is pitch?
Pitch (prosodic cue) is important for turn-taking prediction. Sensitivity analysis confirms the model leverages pitch information, which varies across languages:
- **Mandarin**: Turn-final pitch lowering for all words regardless of lexical tone
- **Japanese**: Turn transition time centers around 0ms
- **English**: More overlaps between turns

### RQ5: Audio encoder effect?
Compared CPC (English pre-trained) with MMS (multilingual wav2vec 2.0). MMS provides benefits for multilingual scenarios but CPC still performs well given its English-only pre-training.

## Cross-linguistic Turn-taking Differences
- Turn transition timing: Mandarin and Japanese ~0ms, English has more overlaps
- Intonation change at end of utterances is effective across all languages
- Backchannel frequency: Japanese highest, then English, then Mandarin
- Both universal tendencies and language-specific differences exist

## Datasets Used
- **English**: Switchboard (telephone conversations)
- **Mandarin**: HKUST/MTS (telephone conversations)
- **Japanese**: CEJC (everyday conversations, with video)

## Relevance to Turn-Taking for Language Learners

- **Critical for L2 turn-taking**: Since monolingual models fail cross-linguistically, a turn-taking system for language learners MUST account for their L1 influence on turn-taking behavior
- A multilingual VAP model could handle learners from different L1 backgrounds with a single model
- L2 speakers may exhibit turn-taking patterns from their L1 (e.g., a Japanese L1 speaker learning English may produce more backchannels than expected)
- The model's ability to identify language could potentially be extended to identify L1 interference in L2 speech
- Pitch sensitivity is important because L2 speakers often transfer L1 prosodic patterns, which could confuse language-specific turn-taking detection
- **For Pipecat fine-tuning**: The VAP architecture (continuous prediction from raw audio using Transformers) is directly relevant as a baseline for adapting turn-taking to L2 Portuguese learners
