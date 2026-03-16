---
title: "Multimodal Self- and Other-Initiated Repairs in L2 Peer Interactions"
authors:
  - Loulou Kosmala
year: 2025
source_url: "https://doi.org/10.21437/DiSS.2025-7"
date_converted: 2026-03-16
---

## Abstract

Examines the multimodal organization of self- and other-initiated repairs in L2 peer interactions between French secondary pupils speaking English. While self-repairs typically reflect errors and language difficulties (syntactic, lexical, phonological), repair mechanisms are not solely responses to errors but tools to negotiate meaning and manage misunderstandings, especially through other-repairs. Adopts a multimodal approach analyzing visual-gestural properties (face, head, hands). Results show a preference for self-repairs over other-repairs, especially those associated with lexical and syntactic difficulties. Published at DiSS 2025 (Disfluency in Spontaneous Speech Workshop), Lisbon.

## Key Findings Relevant to L2 Turn-Taking

### Repair Distribution in L2 Peer Interactions
- Total repairs: **167** (from 12 pupils, 6 pairs)
- **119 self-repairs (71%)** vs. **48 other-repairs (29%)**
- Strong preference for self-repair over other-repair (significant, p = .006)
- Significant individual differences in repair types across speakers

### Self-Repair Types
| Type | Count | Description |
|---|---|---|
| Repetitions | 46 | Most frequent -- repeating same word/phrase |
| Replacements | 29 | Substituting a single word or group |
| False starts | 26 | Cut-off word or sentence |
| Reformulations | 18 | Rephrasing a multi-word unit |

- Significant difference between types (chi-square, p < .05)
- **Lexical** and **syntactic** repairs most frequent (46 and 38 respectively)
- Lexical issues resolved through **repetitions**; phonological issues through **false starts**; syntactic issues through **reformulations** (moderate-to-strong association, Cramer's V = .34)

### Temporal Properties of Self-Repairs (Editing Phase)
- 98 out of 119 self-repairs (82%) contained an editing phase
- **Silent pauses** (SP): 46 instances, mean duration **844ms** (SD=573)
- **Filler particles** (FP): 19 instances -- forms: 24 "euh", 12 "eum", 1 "mm"
- **FP + SP combined**: 33 instances
- Mean FP duration: **522ms** (SD=571)
- Repetitions tend to be more frequently associated with FP+SP combinations (p = .037)
- Pause duration does NOT differ across repair types (non-significant for both FPs and SPs)

### Other-Repair Types
| Type | Count | Description |
|---|---|---|
| Restricted offers | 26 | Seeking confirmation, offering candidate understanding |
| Open repairs | 12 | Signaling misunderstanding without specifying source |
| Restricted requests | 11 | Seeking specification of a specific element |

- Restricted offers most frequent (significant, p = .014)
- Most other-repairs deal with **understanding** (20/48) and **acceptability** (20/48) problems
- Understanding problems trigger open repairs; acceptability problems trigger restricted offers/requests

### Multimodal Patterns
| Feature | Other-Repair | Self-Repair |
|---|---|---|
| Gaze at interlocutor | 91.7% | 39.5% |
| Gaze averted | 2.1% | 29.4% |
| Gaze at paper | 6.3% | 21.8% |
| No gesture | 75.0% | 43.7% |
| Adaptor gesture | 16.7% | 38.7% |
| Smile | 35.4% | 15.1% |
| Neutral face | 33.3% | 58.0% |

- **Self-repairs**: speakers avert gaze (29.4%), look at paper (21.8%), neutral face (58%), more adaptor gestures (38.7%)
- **Other-repairs**: speakers maintain gaze at interlocutor (91.7%), smile more (35.4%), fewer gestures (75% no gesture)
- Distinct multimodal signatures for self vs. other repair

## Specific Hesitation Pattern Data

- L2 pupils: aged 13-16, French secondary school, lower intermediate English
- Speaking time per participant: 1.4 to 3.6 minutes
- Proficiency levels: A (16-20/20), B (12-15), C (8-11), D (<=7)
- Self-repair editing phase silent pauses: mean **844ms** (much longer than native speaker pauses)
- Filler particle forms in French L2 English: "euh" (24), "eum" (12), "mm" (1)
- French fillers used even when speaking English -- strong L1 transfer

## Implications for Turn-Taking Detection in L2 Speech

1. **Repair sequences are NOT turn boundaries**: 82% of self-repairs include an editing phase (silent pause + filler). These pauses (mean 844ms) are well above typical turn-transition gaps (200ms) and silence thresholds (700ms). The model MUST NOT interpret repair pauses as turn completions.

2. **L1 fillers persist in L2**: French learners use "euh" and "eum" even when speaking English. French speakers speaking Portuguese will almost certainly use French filler forms. The model must recognize French hesitation markers in Portuguese speech as floor-holding signals.

3. **Gaze direction distinguishes repair types**: Self-repairs correlate with gaze aversion (looking away/down), while other-repairs correlate with direct gaze. If multimodal features are available, gaze can help distinguish between processing pauses (self-repair, don't interrupt) and comprehension checks (other-repair, response expected).

4. **Repair clusters signal ongoing processing**: Self-repairs frequently combine FP + SP (33/119 cases = 28%), creating pause clusters of 1+ seconds. These are prime false-positive triggers for silence-based turn detectors. The model needs to recognize repair-in-progress patterns.

5. **Other-repairs are turn-relevant**: When a listener initiates an other-repair (especially restricted offers like "you mean X?"), this IS a turn-taking event. The model should recognize other-repair initiations as legitimate turn entries, distinct from interruptions.

6. **Proficiency affects repair patterns**: Lower-proficiency speakers produce more repairs with longer editing phases. The model should expect French speakers with lower Portuguese proficiency to show more and longer repair sequences.
