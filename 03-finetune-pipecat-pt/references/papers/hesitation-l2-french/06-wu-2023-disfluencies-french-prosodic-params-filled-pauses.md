---
title: "Disfluencies in Continuous Speech in French: Prosodic Parameters of Filled Pauses and Vowel Lengthening"
authors:
  - Yaru Wu
  - Ivana Didirkova
  - Anne-Catherine Simon
year: 2023
source_url: null
date_converted: 2026-03-16
---

## Abstract

This study examines prosodic parameters in two types of disfluencies -- vowel lengthenings and filled pauses -- in approximately 2.5 hours of continuous French speech across 11 different speech genres (prepared, semi-prepared, and unprepared). Analyzes mean fundamental frequency (F0), pitch resets, and duration to compare disfluent syllables with surrounding fluent syllables as a function of speech preparation level. Results show that F0 is lower in filled pauses and disfluent vowel lengthenings than in fluent speech, with filled pauses produced at lower F0 than vowel lengthening. Larger pitch resets occur between disfluent units and their preceding contexts.

## Key Findings Relevant to L2 Turn-Taking

### Prosodic Characteristics of French Filled Pauses

#### Fundamental Frequency (F0) Drop
- **Fluent speech**: mean 19.01 ST (SD=5.97)
- **Vowel lengthening**: mean 18.10 ST (SD=6.43) -- ~1 ST drop
- **Filled pauses**: mean 17.90 ST (SD=7.31) -- ~1.1 ST drop
- F0 drop in filled pauses vs. fluent speech:
  - **Females**: 2.12 ST decrease (from 23.66 to 21.54 ST)
  - **Males**: 1.75 ST decrease (from 16.72 to 14.97 ST)
- Both disfluency types significantly lower than fluent speech (p < 0.001)

#### F0 and Speech Preparation Level
- **Unprepared speech**: FP average F0 = 18.89 ST
- **Semi-prepared speech**: FP average F0 = 17.41 ST
- **Prepared speech**: FP average F0 = 14.24 ST
- Massive **4.66 ST drop** between unprepared and prepared speech for FPs
- Vowel lengthening does NOT follow this trend (only 0.07 ST difference)
- Fluent speech shows moderate 1.2 ST drop between unprepared and prepared

#### Pitch Reset (Melodic Discontinuity)
- Filled pauses create a **larger negative pitch reset** from the preceding syllable than vowel lengthening
- Both FPs and lengthening produce significantly more negative pitch changes than fluent speech (p < 0.001)
- No significant pitch reset difference between the disfluent unit and the FOLLOWING syllable
- Key insight: the pitch drop **before** a filled pause is a reliable acoustic cue

#### Duration
- **Vowel lengthening** is significantly LONGER than **filled pauses** (p < 0.001)
- Vowel lengthening mean: ~347ms; Filled pause mean: ~268ms (from prior French study by Grosman 2018)
- Filled pauses are shorter in prepared speech and longer in unprepared speech
- Vowel lengthening shows the opposite pattern: shorter in unprepared, longer in prepared/semi-prepared

### Corpus Details (LOCAS-F)
- Multi-genre French oral corpus (14 speech genres)
- Analyzed: 2h38m of recordings
- Data: >1,500 vowel prolongation sequences, >1,000 filled pauses, ~59,000 fluent syllables
- Disfluency affects ~4% of all data (typical for non-pathological speech)
- Inter-annotator agreement: kappa 0.86 for FPs (near perfect), 0.64 for lengthening (substantial)

## Specific Hesitation Pattern Data

- French filled pauses typically transcribed as **"euh"**
- Duration range for hesitation vowels (cross-linguistic): **200ms to 650ms** (from Vasilescu et al. 2004, 8 languages)
- French FP F0 value is similar to the **onset value of breath groups** for a given speaker
- Mean FP duration in French: **268.4ms** (Grosman 2018)
- Mean vowel lengthening duration in French: **347.25ms** (Grosman 2018)

## Implications for Turn-Taking Detection in L2 Speech

1. **F0 drop is a reliable filled pause detector**: Filled pauses in French show a consistent 1-2 ST drop in F0 compared to fluent speech. A turn-taking model can use this prosodic signature to identify FPs even without lexical recognition. This is especially useful when ASR may not reliably transcribe L2 "euh" sounds.

2. **Pitch reset before FPs marks processing onset**: The significant negative pitch reset between the preceding syllable and the filled pause signals the START of a hesitation event. This could serve as an early warning that the speaker is about to hesitate but intends to continue.

3. **Preparation level affects FP prosody**: In more spontaneous speech (like real-time conversation), FPs have HIGHER F0 (closer to fluent speech). In prepared speech, FPs drop dramatically in F0. Since L2 speakers in conversation are in "unprepared" mode, their FPs may be harder to distinguish from fluent speech by F0 alone.

4. **Vowel lengthening is a separate hesitation signal**: French speakers (both L1 and L2) elongate vowels at word boundaries as a hesitation strategy. These are LONGER than filled pauses (~347ms vs. ~268ms) but show a smaller F0 drop. The model should recognize elongated schwa-like sounds as hesitation, not turn completion.

5. **4% disfluency rate in native speech**: This baseline helps calibrate expectations. L2 speakers will show significantly higher rates, potentially 8-15% or more, meaning the model will encounter disfluency markers very frequently in L2 French-to-Portuguese speech.

6. **Cross-linguistic prosodic transfer**: French speakers speaking Portuguese will likely transfer their French FP prosodic patterns (F0 drop magnitude, pitch reset patterns). The model should accommodate French-influenced prosodic contours on Portuguese hesitations.
