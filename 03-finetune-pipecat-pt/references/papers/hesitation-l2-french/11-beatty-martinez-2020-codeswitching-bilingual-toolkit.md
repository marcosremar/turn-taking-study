---
title: "Codeswitching: A Bilingual Toolkit for Opportunistic Speech Planning"
authors:
  - Anne L. Beatty-Martinez
  - Christian A. Navarro-Torres
  - Paola E. Dussias
year: 2020
source_url: "https://doi.org/10.3389/fpsyg.2020.01699"
date_converted: 2026-03-16
---

## Abstract

Reviews codeswitching as a bilingual strategy for opportunistic speech planning. Recent discoveries show that codeswitching is not haphazard but subject to unique linguistic and cognitive constraints, and that bilinguals who codeswitch exhibit usage patterns conforming to community-based norms. The paper provides corpus evidence (from the Puerto Rico Codeswitching Map Task corpus of Spanish-English bilinguals) that codeswitching serves as a tool to navigate linguistic interference during production, enabling speakers to circumvent speech planning difficulties by opportunistically drawing from whichever language is most active or accessible.

## Key Findings Relevant to L2 Turn-Taking

### Codeswitching as a Fluency Strategy
- Codeswitching is NOT random -- it is **structured and strategic**
- Functions as a tool for **opportunistic speech planning**: bilinguals take advantage of whichever language's words/structures are most active to achieve communicative goals
- Reduces the cost in time and resources during speech production
- **93% of codeswitches occur at Intonation Unit (IU) boundaries** (Plaistowe 2015, from NMSEB corpus)
- This means switches overwhelmingly happen at natural prosodic break points, not mid-phrase

### Prosodic Signatures of Codeswitching
- Speech rate changes before codeswitches: momentary **reorganization of prosodic and phonetic systems**
- These changes serve a dual purpose:
  1. Help the speaker negotiate lexical competition and minimize cross-language interference
  2. Provide reliable **acoustic cues for listeners** to anticipate and process the switch
- Codeswitching affects bilingual speech at different levels: word-level (speech rate of individual words) and sentence-level (speech rate within prosodic sentences)

### Language Control Modes
- **Single-language context**: Language control is COMPETITIVE -- one language suppressed at expense of other
- **Codeswitching context**: Language control is COOPERATIVE -- coactivation maintained, items from both languages available for selection
- Dense codeswitchers minimize language membership tagging and keep both languages active

### Corpus Data (Puerto Rico Codeswitching Map Task)
- 10 Spanish-English bilinguals (6 female), all native Spanish speakers
- Equal self-reported proficiency in both languages (9.6/10 each)
- ~2.5 hours of unscripted, task-oriented dialogs
- Participants exposed to both languages: more Spanish with family, more English in media, equal among friends
- Paired with in-group confederate (close friend from same speech community) -- this increased codeswitching 4x vs. out-group pairing

### Variable Equivalence and Switch Sites
- Bilinguals do NOT consistently avoid "conflict sites" between languages
- Instead, they **opportunistically use** sites of variable equivalence (partial structural overlap between languages)
- This challenges the strict "equivalence constraint" (Poplack 1980)

## Specific Data Points

- Self-reported proficiency: Spanish 9.6/10 (SD=0.8), English 9.6/10 (SD=0.5)
- Mean age: 23.3 years (SD=1.8)
- Codeswitching at IU boundaries: 93% (from related corpus study)

## Implications for Turn-Taking Detection in L2 Speech

1. **Codeswitches happen at prosodic boundaries**: Since 93% of codeswitches align with intonation unit boundaries, a turn-taking model should expect language switches at natural break points -- the same locations where turn transitions might occur. The model must not interpret a language switch as a turn-ending signal.

2. **Prosodic reorganization before switches**: The speech rate and prosodic changes before a codeswitch could be confused with turn-ending prosody. For French speakers occasionally inserting French words while speaking Portuguese, the model needs to tolerate these prosodic perturbations without triggering a false turn-shift prediction.

3. **L1-L2 fluency trade-off**: When French speakers struggle with Portuguese lexical retrieval, they may briefly switch to French (a natural bilingual strategy). The turn-taking model should recognize this as a hold signal (the speaker is using an alternative strategy to continue) rather than a breakdown signal.

4. **Cooperative language mode in bilingual contexts**: If the conversation partner also speaks French, the French-Portuguese speaker may engage in cooperative codeswitching. The model should handle mixed-language utterances without treating language boundaries as turn boundaries.

5. **Community norms matter**: Codeswitching frequency and patterns depend heavily on the interactional context and community norms. The model should be configurable for different bilingual settings (e.g., high-codeswitch vs. monolingual-target contexts).

6. **Intonation unit alignment**: Since codeswitches cluster at IU boundaries, and IU boundaries are also key sites for turn-taking decisions, the model must weigh additional cues (gaze, content completeness, prosodic finality) at these ambiguous boundary points where both a codeswitch-and-continue and a turn-yield are possible.
