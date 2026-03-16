---
title: "The Dual Status of Filled Pauses: Evidence from Genre, Proficiency and Co-occurrence"
authors:
  - Loulou Kosmala
  - Ludivine Crible
year: 2022
source_url: "https://doi.org/10.1177/00238309211010862"
date_converted: 2026-03-16
---

## Abstract

A corpus study examining the lexical vs. non-lexical status of filled pauses ("euh" and "eum") in spoken French. Analyzes their distribution across communication settings (prepared monologues vs. spontaneous conversations) and language proficiency levels (native vs. non-native French). Quantitative findings reveal differences in frequency, duration, position, and co-occurrence patterns. Qualitative analysis identifies two distinct patterns: (1) initial position clustered with a discourse marker (fluent, structuring use) vs. (2) medial position clustered with other hesitation markers (disfluent, processing use). The authors argue for a **dual status** of filled pauses based on formal, functional, and contextual features.

## Key Findings Relevant to L2 Turn-Taking

### French Filled Pause Forms
- Two main variants in French: **"euh"** [schwa] and **"eum"** [nasal]
- "Eum" is associated with longer delays and major discourse transitions/boundaries
- "Euh" is the dominant form in both native and non-native French (84-92% of all FPs)

### Genre Effects (DisReg Corpus -- French Native Speakers)
- **Class presentations**: 6.8 FPs per 100 words (mean duration 415ms)
- **Casual conversations**: 4.2 FPs per 100 words (mean duration 343ms)
- More filled pauses in monologues than dialogues (significant, LL = 47.02, p < .001)
- More "eum" in presentations (23%) vs. conversations (8%)
- More initial-position FPs in presentations (40%) vs. conversations (24%)
- More final-position FPs in conversations (14%) vs. presentations (3%) -- reflects **turn-yielding** function

### Native vs. Non-Native French (SITAF Corpus)
- **Native rate**: 4.4 FPs per 100 words (mean duration **378ms**)
- **Non-native rate**: 5.3 FPs per 100 words (mean duration **524ms**)
- Rate difference not significant, but **duration difference highly significant** (p < .001)
- Both groups: ~60% medial position, ~30% initial, ~10% final
- Non-native speakers had more **standalone and interrupted** positions (exclusive to learners)
- Non-native speakers cluster FPs with other fluencemes more often (82% clustered vs. 72% for natives)
- Longer fluenceme sequences in learner speech signal higher disruption

### Co-occurrence with Discourse Markers
- 69 instances of FP + discourse marker clusters found
- Common French discourse markers paired with FPs: "donc" (so), "mais" (but), "ben" (well), "alors" (then/well), "en fait" (actually), "enfin" (I mean)
- FP position shifts when clustered with discourse markers: **57% initial** (vs. 73% medial when isolated)
- **Two distinct patterns emerge**:
  1. **Fluent pattern**: FP + discourse marker at turn/phrase boundary (initial position) -- structuring function
  2. **Disfluent pattern**: FP in medial position clustered with repetitions, lengthenings, silent pauses -- processing difficulty

### Functional Analysis of Discourse Marker + FP Clusters
Four discourse domains identified:
- **Ideational**: connecting facts (e.g., "alors euh" = "then euh")
- **Rhetorical**: expressing opinions (e.g., "mais euh" = "but euh")
- **Sequential**: marking transitions (e.g., "donc euh" = "so euh")
- **Interpersonal**: monitoring/agreement (e.g., "bah oui euh" = "well yes euh")

## Specific Hesitation Pattern Data

- Native French FP mean duration: **378ms** (SD=200ms)
- Non-native French FP mean duration: **524ms** (SD=222ms) -- 146ms longer
- FP duration is a more reliable index of proficiency than frequency
- Duration of FPs with/without discourse markers: no significant difference (~433ms vs. ~467ms)
- Example extreme disfluent sequence from a learner: 8 fluencemes in a row (lengthening + 3 silent pauses + 3 filled pauses + self-interruption), with pauses up to **1,177ms**

## Implications for Turn-Taking Detection in L2 Speech

1. **Dual function detection is essential**: The same sound "euh" can signal either (a) turn/phrase structuring (fluent, initial position + discourse marker) or (b) processing difficulty (disfluent, medial position + hesitation cluster). Both mean "don't take the turn yet" but for different reasons.

2. **Duration as proficiency proxy**: Non-native FPs are ~150ms longer on average. A model trained on native French data will encounter systematically longer hesitations from L2 speakers. This duration difference should NOT be interpreted as turn-yielding.

3. **Final-position FPs signal turn-yielding**: In conversations, 14% of FPs occur in final position (vs. 3% in monologues). The combination of FP + silent pause at utterance end is a reliable turn-yielding signal (e.g., "but he has a role euh (0.924)...").

4. **Discourse marker + FP combinations**: Patterns like "donc euh" (so euh), "mais euh" (but euh) are strong turn-HOLD signals in French. A model should recognize these common French fluenceme clusters.

5. **Cluster length matters**: Isolated FPs may go unnoticed; clustered FPs with repetitions and silent pauses signal genuine difficulty. Longer clusters = speaker is struggling but still holding the floor.

6. **L2 speakers underuse discourse markers**: Non-native speakers rely more heavily on filled pauses alone, whereas native speakers combine FPs with rich discourse markers. The model should expect L2 French-Portuguese speakers to show heavy FP reliance with fewer Portuguese discourse markers.
