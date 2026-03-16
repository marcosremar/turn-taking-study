---
title: "Exploring the Effects of Bilingualism on Filled Pauses: An Acoustic-Phonetic Perspective"
authors:
  - Justin Jing Hoi Lo
year: 2018
source_url: null
date_converted: 2026-03-16
---

## Abstract

Examines the phonetic realization of filled pauses (FPs) in bilingual speakers from an acoustic-phonetic perspective. Focuses on German-French simultaneous bilinguals to explore how knowledge of multiple languages influences the production of FPs. Addresses two key questions: (1) whether bilinguals differentiate FPs across their two languages, and (2) whether bilingual FPs are acoustically distinct from monolingual FPs in the same language. Data from 15 female German-French simultaneous bilinguals (HABLA corpus) compared with 20 female French monolinguals (NCCFr corpus).

## Key Findings Relevant to L2 Turn-Taking

### Language-Specific Filler Phonetics
- **Bilinguals differentiate their FPs by language**: German vs. French FPs from the same speakers are acoustically distinct
- German FPs ("ah"): **shorter duration**, **lower F1-F3 formants**
- French FPs ("euh"): **longer duration**, **higher F1-F3 formants**
- This demonstrates that filled pauses are NOT universal grunts but have **language-specific phonetic targets**

### Bilingual vs. Monolingual French FPs
- French FPs differed between bilinguals and monolinguals **only in duration** (not formant structure)
- Vocalic quality (F1-F3) was similar between bilingual and monolingual French FPs
- The formant structure of "euh" is maintained even by bilinguals -- it is a stable phonetic category

### Methodological Details
- Analyzed FP variant: **UH type** only ("euh" in French, "ah" in German) -- the non-nasal variant
- Measurements: midpoint frequencies of F1, F2, F3 and duration from each vocalic segment
- Data: spontaneous speech recordings in both languages from the same speakers
- Bilingual speakers: 15 female, simultaneous acquisition of both German and French from birth
- Monolingual speakers: 20 female French monolinguals (NCCFr corpus, Torreira et al. 2010)

### Cross-Linguistic Patterns
- Prior research (Candea et al. 2005) established language-specific patterns in vocalic quality of FPs across languages
- Speakers also exhibit **personal, speaker-specific FP variants** (Kunzel 1997)
- For bilinguals, both language-specificity and speaker-specificity are **intertwined** -- each speaker has distinct FPs per language

## Specific Hesitation Pattern Data

- French "euh": central vowel [schwa], longer, higher formants
- German "ah": open vowel, shorter, lower formants
- FP duration difference is the primary distinguishing feature between bilingual and monolingual productions
- Speaker-specific variation exists within the language-specific norms

## Implications for Turn-Taking Detection in L2 Speech

1. **French "euh" has a specific acoustic signature**: The formant structure (F1, F2, F3) of French filled pauses is distinct from other languages' fillers. A turn-taking model processing French speakers' Portuguese should be trained to recognize the French "euh" spectral pattern, which will likely persist even when the speaker is producing Portuguese.

2. **Duration is the bilingual marker**: Since bilingual FPs differ from monolingual FPs primarily in duration (not formant quality), the model should expect French-Portuguese bilinguals to produce French-quality "euh" sounds but with potentially different durations than monolingual French speakers.

3. **Language-specific FPs are real lexical items**: The fact that simultaneous bilinguals maintain distinct FP phonetics across languages supports treating FPs as language-specific "words" rather than involuntary sounds. This means the FP form can reveal which language is being processed, useful for detecting language switching.

4. **Formant-based FP detection**: Since F1-F3 formants reliably distinguish French vs. German FPs, formant analysis could help a model determine whether a bilingual speaker is in "French processing mode" or "Portuguese processing mode" based on their filler pronunciation. This could improve turn-taking prediction by identifying the currently active language.

5. **Speaker adaptation is needed**: Speaker-specific FP variants mean the model benefits from per-speaker calibration of what counts as a filled pause. During an initial calibration period, the model could learn each speaker's characteristic FP formant patterns and durations.

6. **Simultaneous bilinguals maintain separation**: Even speakers who acquired both languages from birth maintain distinct FP phonetics. This means L2 speakers (sequential bilinguals) will likely show even MORE pronounced L1 influence on their L2 fillers, making French "euh" recognition in Portuguese speech essential.
