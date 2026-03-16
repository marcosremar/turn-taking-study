---
title: "A Finite-State Turn-Taking Model for Spoken Dialog Systems"
authors:
  - Antoine Raux
  - Maxine Eskenazi
year: 2009
source: https://aclanthology.org/N09-1071/
date_converted: 2026-03-16
---

## Abstract

This paper introduces the Finite-State Turn-Taking Machine (FSTTM), a new model to control the turn-taking behavior of conversational agents. Based on a non-deterministic finite-state machine, the FSTTM uses a cost matrix and decision-theoretic principles to select a turn-taking action at any time. The authors show how the model can be applied to the problem of end-of-turn detection. Evaluation results on a deployed spoken dialog system show that the FSTTM provides significantly higher responsiveness than previous approaches.

**Note**: The PDF file `finite_state_turn_taking_raux_2009.pdf` in this directory is corrupted (contains an unrelated physics paper). Content for this summary was sourced from the actual paper PDF downloaded from ACL Anthology.

## Key Contributions

1. **FSTTM model**: A principled, unified framework for turn-taking control based on a 6-state non-deterministic finite-state machine with decision-theoretic action selection.
2. **Data-driven optimization**: Unlike prior hand-coded models, the FSTTM's cost parameters can be optimized from data using logistic regression.
3. **Anytime endpointing**: The model can make end-of-turn decisions both during pauses and during speech (before a pause is even detected by VAD), enabling faster response times.
4. **Deployed evaluation**: Tested on a real, publicly deployed bus information system, not just in lab conditions.

## Architecture / Method Details

### Six-State Finite-State Model

Based on Jaffe & Feldstein (1970) and Brady (1969), the FSTTM uses six states representing who holds the conversational floor:

| State | Description |
|-------|-------------|
| **USER** | User has the floor (obligation or intention to speak) |
| **SYSTEM** | System has the floor |
| **FREES** | Floor is free, following a SYSTEM state |
| **FREEU** | Floor is free, following a USER state |
| **BOTHS** | Both claim the floor, following a SYSTEM state |
| **BOTHU** | Both claim the floor, following a USER state |

States are defined in terms of **intentions and obligations**, not surface-level speech/silence observations. For example, USER state persists during mid-turn pauses.

### Four Turn-Taking Actions

At any time, the system can take one of four actions:
- **Grab (G)**: Claim the floor
- **Release (R)**: Relinquish the floor
- **Keep (K)**: Maintain floor claim
- **Wait (W)**: Remain silent without claiming floor

### Turn-Taking Phenomena Captured

The model formalizes common turn-taking patterns as 2-step state transitions:

1. **Turn transitions with gap**: `SYSTEM --(R,W)--> FREES --(W,G)--> USER` (most common)
2. **Turn transitions with overlap**: `SYSTEM --(K,G)--> BOTHS --(R,K)--> USER` (barge-in)
3. **Failed interruptions**: `USER --(G,K)--> BOTHU --(R,K)--> USER` (system interrupts then backs off)
4. **Time outs**: `SYSTEM --(R,W)--> FREES --(G,W)--> SYSTEM` (user doesn't respond)

### Decision-Theoretic Action Selection

The optimal action minimizes expected cost:

```
C(A) = sum over S in States: P(s=S|O) * C(A, S)
```

Where `P(s=S|O)` is estimated from observable features and `C(A,S)` comes from a cost matrix.

### Cost Matrix Design Principles

Derived from Sacks et al. (1974) -- "participants minimize gaps and overlaps":

1. Actions that resolve gaps/overlaps have zero cost.
2. Actions that create unwanted gaps/overlaps have a constant cost parameter.
3. Actions that maintain gaps/overlaps have cost proportional to time in that state.

Four cost parameters:
- **CS**: Cost of false interruption (interrupting system prompt when user is not claiming floor)
- **CO(tau)**: Cost of remaining in overlap for tau ms
- **CU**: Cost of cut-in (grabbing floor when user holds it)
- **CG(tau)**: Cost of remaining in a gap for tau ms (set as CG^p * tau)

### Probability Estimation

The key estimation task is `P(FREEU | Ot)` -- the probability that the user has released the floor.

**At pauses** (VAD detects silence): Uses Bayes rule combining:
- `P(F|O)`: Prior probability of floor release from pause-onset features (logistic regression)
- `P(d >= tau | O, U)`: Probability pause lasts at least tau ms given user holds floor (exponential distribution)

**During speech** (before VAD pause detection): Separate logistic regression on each ASR partial hypothesis, enabling endpointing before a full pause is detected.

### Features Used

All features are automatically extractable at runtime:
- **Dialog state**: Open question / Closed question / Confirmation
- **Turn-taking features**: Whether current utterance is a barge-in
- **Semantic features**: From dialog state and partial ASR hypotheses
- **Boundary LM score**: Ratio of log-likelihood of hypothesis being complete vs. incomplete (trained on ASR output, no human transcription needed). **Most informative feature across all states.**
- **Average words per utterance** so far
- **Confirmation markers**: "YES", "SURE", etc.
- **Pause duration** within partial hypothesis (0-200ms range)

## Experimental Results

### Logistic Regression for P(F|O)

Trained with stepwise regression, 10-fold cross-validation on 586 dialogs (2008 corpus):

| Dialog State | Pause: Class. Error | Pause: Log-Likelihood | Speech: Class. Error | Speech: Log-Likelihood |
|---|---|---|---|---|
| Open question | 35% (vs 38% baseline) | -0.61 (vs -0.66) | 17% (vs 20%) | -0.40 (vs -0.50) |
| Closed question | 26% (vs 25% baseline) | -0.50 (vs -0.56) | 22% (vs 32%) | -0.49 (vs -0.63) |
| Confirmation | 12% (vs 12% baseline) | -0.30 (vs -0.36) | 17% (vs 36%) | -0.40 (vs -0.65) |

The "in speech" model achieves much larger gains, especially for Closed questions and Confirmations -- classification error drops from 32% to 22% and 36% to 17% respectively.

### Batch Evaluation: Latency vs. Cut-in Rate

**In-pause FSTTM vs. baselines** (2007 corpus):
- FSTTM outperforms fixed threshold baseline by up to **29.5% latency reduction**.
- Slight improvement over Ferrer et al. (2003) reimplementation.

**Anytime-FSTTM vs. in-pause FSTTM** (2008 corpus):
- At 5% cut-in rate: anytime-FSTTM yields latencies **17% shorter** than in-pause-FSTTM, and **40% shorter** than fixed threshold baseline.
- **30-40% of turns are endpointed before the pause is detected by VAD** (during speech).

### Live Evaluation (Deployed System)

A/B test over 10 days: 171 FSTTM dialogs vs. 148 fixed-threshold control dialogs.

Settings: `CG^p = 1, CG^s = 500, CU = 5000` (FSTTM) vs. 555ms fixed threshold (control). Both calibrated for ~6.3% cut-in rate.

| Metric | FSTTM | Fixed Threshold |
|--------|-------|-----------------|
| Average latency | **320ms** | 513ms |
| Cut-in rate | **4.8%** | 6.3% |

Results:
- **193ms latency reduction** (p < 0.05, statistically significant).
- 1.5% cut-in rate reduction (not statistically significant, but directionally correct).

## Relevance to Turn-Taking / End-of-Turn Detection

The FSTTM is a foundational reference for BabelCast's end-of-turn detection:

1. **Decision-theoretic framework**: The cost-based approach directly applies to our system. We can define costs for premature LLM invocation (wasted compute, incorrect partial translation) vs. delayed response (poor user experience), and optimize the trade-off.

2. **Anytime endpointing**: The insight that 30-40% of turns can be endpointed *before the pause is detected by VAD* is powerful. In our pipeline, this means we could start the translation process before the speaker's pause is even fully formed, dramatically reducing perceived latency.

3. **Boundary LM score**: The most informative feature in Raux's system was a language model score measuring syntactic completeness. We already have ASR partial results from Whisper -- we could compute a similar feature to predict turn completeness without waiting for silence.

4. **Cost parameter tuning**: The four-parameter cost matrix (CS, CO, CU, CG) provides a compact, interpretable way to tune system behavior. We could expose these as configuration parameters in our Pipecat pipeline, allowing adjustment of responsiveness vs. interruption rate.

5. **Real-world validation**: This is one of the few ETD papers validated on a deployed system with real users, not just lab data. The 193ms latency improvement is a concrete, practically meaningful result.

6. **Simplicity**: The FSTTM is mathematically elegant and computationally trivial -- just a few probability estimates and a cost comparison. It could be implemented alongside our Silero VAD as an additional decision layer with negligible overhead.

7. **Limitation for our case**: The system was designed for a task-oriented telephone dialog (bus information), which has more predictable turn structures than the open-domain meeting conversations we handle. The boundary LM approach would need adaptation for our more varied content.
