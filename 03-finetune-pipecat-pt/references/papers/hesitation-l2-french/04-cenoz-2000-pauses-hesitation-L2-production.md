---
title: "Pauses and Communication Strategies in Second Language Speech"
authors:
  - Jasone Cenoz
year: 2000
source_url: "https://eric.ed.gov/?id=ED426630"
date_converted: 2026-03-16
---

## Abstract

A study of silent and filled pauses in second language speech analyzing (1) which types of pauses are produced, (2) the functions of non-juncture pauses, (3) whether pauses co-occur with other hesitation phenomena, and (4) whether the occurrence of pauses is associated with second language proficiency. Subjects were 15 intermediate and advanced learners of English as a second language (L1 Spanish, university students). Each told a story in English which was recorded and transcribed. Silent and filled pauses at non-grammatical junctures were identified and analyzed.

## Key Findings Relevant to L2 Turn-Taking

### Pause Distribution
- Total non-juncture pauses: **1,085** across 15 subjects
- **64% silent pauses**, **36% filled pauses**
- The most common filler was **"eh"** (transferred from L1 Spanish), demonstrating L1 transfer of hesitation markers
- Wide individual variation: filled pauses ranged from 4% to 74.5% of all pauses per individual

### Silent Pause Durations
| Duration Range | % of Pauses | Subjects Affected | Individual Variation |
|---|---|---|---|
| 200-1000ms | 70% | 100% | 39%-96% |
| 1001-2000ms | 21% | 100% | 4%-39% |
| 2001-4000ms | 7% | 40% | 10%-18% |
| 4001ms+ | 2% | 40% | 1.5%-7% |

### Functional Categories of Pauses
| Function | Silent Pauses | Filled Pauses |
|---|---|---|
| Lexical (retrieval) | 36% | 26% |
| Morphological | 5% | 1% |
| Planning | 59% | 73% |

- Both silent and filled pauses serve the same functions, but filled pauses are overwhelmingly used for general **planning** (73%)
- More silent pauses associated with **lexical retrieval** than filled pauses

### Co-occurrence with Other Hesitation Phenomena
| Strategy | Silent Pauses | Filled Pauses |
|---|---|---|
| Pauses + other hesitations | 54% | 23% |
| Only pauses | 46% | 77% |

- Most common hesitation phenomena: **repetition, self-correction, and reformulation**
- Silent pauses much more likely to co-occur with other repair strategies (54%) vs. filled pauses (23%)
- Filled pauses function as **standalone repair devices**; silent pauses tend to **precede** other repairs

### Proficiency Effects
- Higher-proficiency learners produced **more total pauses** and **more filled pauses** (53% silent + 46% filled)
- Lower-proficiency learners: 69% silent + 31% filled
- Lower-proficiency learners used **more hesitation strategies combined with pauses** (62% for silent pauses) vs. higher-proficiency (38%)
- Interpretation: high-proficiency learners need only time to retrieve information; low-proficiency learners need to vocalize different options

## Specific Hesitation Pattern Data

- Silent pause minimum threshold: **200ms**
- Pause range: **205ms to 11,569ms**
- Most pauses (70%) are under 1 second
- Long pauses (>2s) only in 40% of subjects -- marks a subset of struggling speakers
- L1 Spanish filler "eh" dominates filled pauses (L1 transfer effect)

## Implications for Turn-Taking Detection in L2 Speech

1. **Pause duration is critical**: 70% of L2 pauses are 200-1000ms, overlapping with natural turn-transition gaps (~200ms). A turn-taking model must distinguish these within-turn hesitation pauses from actual turn boundaries.

2. **Filled pauses signal continuation**: Since filled pauses are primarily standalone floor-holding devices (77% occur without other hesitations), detecting "eh/euh/um" should strongly indicate the speaker intends to continue, NOT yield the turn.

3. **L1 transfer of filler forms**: French speakers speaking Portuguese will likely transfer French "euh" rather than adopting Portuguese fillers. The model should recognize L1-influenced fillers as valid hold signals.

4. **Proficiency paradox**: More proficient L2 speakers use MORE filled pauses, not fewer. The model should not interpret high filler rates as low fluency or turn-yielding intent.

5. **Cluster detection**: When silent pauses co-occur with repetitions, self-corrections, or reformulations (54% of cases), this strongly signals ongoing speech processing, not turn completion. Detecting hesitation clusters is key to avoiding premature turn-taking.

6. **Individual variation is massive**: Some speakers use almost no filled pauses (4%) while others fill 74.5% of their pauses. The model needs speaker adaptation or robust handling of diverse hesitation profiles.
