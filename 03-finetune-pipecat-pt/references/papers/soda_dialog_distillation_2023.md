---
title: "SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization"
authors:
  - Hyunwoo Kim
  - Jack Hessel
  - Liwei Jiang
  - Peter West
  - Ximing Lu
  - Youngjae Yu
  - Pei Zhou
  - Ronan Le Bras
  - Malihe Alikhani
  - Gunhee Kim
  - Maarten Sap
  - Yejin Choi
year: 2023
source: https://arxiv.org/abs/2212.10465
date_converted: 2026-03-16
---

## Abstract

Data scarcity has been a long standing issue in the field of open-domain social dialogue. SODA (SOcial DiAlogues) is the first publicly available, million-scale high-quality social dialogue dataset. By contextualizing social commonsense knowledge from a knowledge graph, the authors distill an exceptionally broad spectrum of social interactions from a large language model (GPT-3.5). Human evaluation shows that conversations in SODA are more consistent, specific, and (surprisingly) natural than those in prior human-authored datasets. Using SODA, they train COSMO, a generalizable conversation model that significantly outperforms best-performing conversation models (GODEL, BlenderBot-1, Koala, Vicuna) on naturalness and consistency in unseen datasets. COSMO responses are even sometimes preferred over original human-written gold responses.

## Key Contributions

1. **SODA dataset**: 1.5 million dialogues, 11 million utterances, 300 million tokens -- the largest publicly available open-domain social conversation dataset (CC-BY-4.0 license).
2. **CO3 framework**: COntextualizing COmmonsense for distilling COnversations -- a pipeline that transforms commonsense knowledge triples into narratives, then into dialogues via LLM.
3. **COSMO model**: A conversation model trained on SODA that generalizes better than existing models to unseen datasets.
4. **Insight on naturalness**: LLM-based agents (Koala, Vicuna, ChatGPT) tend to generate informative but unnatural responses in social chitchat contexts.

## Method Details

### CO3 Framework (Conversation Distillation Pipeline)

Three-step process:

1. **Commonsense Knowledge Retrieval**: Sample social commonsense triples from Atomic10x knowledge graph. Example: `(Head: PersonX moves a step closer to the goal, Relation: xNeed, Tail: to take the first step)`.

2. **Knowledge to Narrative**: Convert triples to sentence form, then prompt GPT-3.5 to generate a 2-3 sentence narrative contextualizing the commonsense. Replace person variables with common names.

3. **Narrative to Conversation**: Infer conversation participants from the narrative (via GPT-3.5), then generate a full multi-turn dialogue grounded in the narrative.

### Dataset Construction and Filtering

Starting from 2.2M initial conversations from GPT-3.5:
- Lexical pattern matching for erroneous patterns (6.3% removed)
- Turn count filter: keep 4-20 turns only (5.7% removed)
- Speaker count filter: max 2 speakers (11.3% removed)
- Non-human speaker filter (5.6% removed)
- Safety filtering: Canary model + Rewire API for violence/hate/explicit content (~5.3% removed)
- Commonsense filtering: GPT-3.5 zero-shot classifier verifies head event is present (95% pass)
- Name bias mitigation: Random replacement with Top-10K US SSN names

**Final dataset: 1,486,896 conversations (68.9% retention)**

### Dataset Statistics

| Dataset | # Dialogues | Avg Turns | Avg Utt Length | Lexical Diversity (MTLD) |
|---------|-------------|-----------|----------------|--------------------------|
| DailyDialog | 13K | 7.9 | 14.6 | 63.0 |
| PersonaChat | 11K | 14.8 | 14.2 | 43.6 |
| WizardOfWikipedia | 22K | 9.1 | 16.4 | 60.3 |
| EmpatheticDialogue | 25K | 4.3 | 13.7 | 64.2 |
| BlendedSkillTalk | 7K | 11.2 | 13.6 | 64.2 |
| ProsocialDialog | 58K | 5.7 | 20.0 | 60.2 |
| **SODA** | **1.5M** | **7.6** | **16.1** | **68.0** |

SODA is 100x larger than previous datasets and has the highest lexical diversity.

### Topic Distribution (from Atomic10x relations)

| Relation | % of SODA | Top Keywords |
|----------|-----------|--------------|
| xAttr (18%) | | kindness, anger, intelligent, responsibility |
| xEffect (17%) | | gratitude, anger, upset, hard work |
| xIntent (23%) | | independence, hard work, determination |
| xNeed (7%) | | job, money, confidence, comfort |
| xReact (25%) | | frustration, anger, confidence, happy |
| xWant (11%) | | conversation, store, determination |

Includes 385K conversations with rich emotional content from 1.7K unique emotion descriptions.

## Experimental Results

### Human Evaluation: SODA vs. Human-Authored Datasets

Head-to-head comparisons on 300 dialogues each, 6 criteria:

**SODA vs. DailyDialog**: SODA preferred on all 6 axes by a large margin (statistically significant, |z| > 3.3, p < 0.05).

**SODA vs. BlendedSkillTalk**: SODA preferred on 5 of 6 axes (all except Context Dependence).

Evaluated criteria: natural flow, context dependence, topic consistency, speaker consistency, specificity, overall quality.

### COSMO Model Performance

COSMO (trained on SODA with LLaMA backbone) vs. existing models in head-to-head human evaluation:

- **COSMO vs. BlenderBot**: COSMO wins by >40% average across comparisons.
- **COSMO vs. Koala**: COSMO wins by >40% average.
- **COSMO vs. Vicuna**: COSMO wins by >40% average.
- **COSMO vs. GODEL**: COSMO significantly preferred.

Notable finding: COSMO outperforms BlenderBot on BlenderBot's own training data (BlendedSkillTalk), despite never seeing that corpus. COSMO responses are even preferred over human-authored ground-truth responses in DailyDialog.

### Insight on LLM Naturalness

LLM-based agents (Koala, Vicuna, ChatGPT) generate informative but unnatural responses in social chitchat -- they tend to provide knowledge-based answers rather than natural conversational replies. SODA and COSMO highlight this distinction between knowledge-enriched conversation and natural social dialogue.

## Relevance to Turn-Taking / End-of-Turn Detection

SODA's relevance to BabelCast's turn-taking work is indirect but valuable:

1. **Training data for ETD models**: SODA's 1.5M dialogues with natural turn structures can serve as a source of text-based dialogue data for creating synthetic speech datasets for end-of-turn detection (similar to how SpeculativeETD used MultiWOZ + TTS). The diverse social interactions in SODA would produce more varied turn-taking patterns than task-oriented datasets.

2. **Turn structure diversity**: With 7.6 average turns per conversation and diverse topics, SODA provides examples of many different turn-taking patterns -- short exchanges, long monologues, emotional conversations, etc. -- which is important for training robust ETD models.

3. **Pause and hesitation modeling**: The CO3 framework's approach of inserting filler words and pauses into synthetic data (as done in SpeculativeETD with MultiWOZ) could be applied to SODA at 100x the scale, creating a much larger and more diverse ETD training set.

4. **Conversation context for linguistic ETD features**: If our ETD model uses linguistic features (dialogue act, semantic completeness), SODA's grounded narratives provide context that could help train models to understand when a conversational point has been completed vs. when more is expected.

5. **Naturalness benchmark**: The finding that LLM responses lack social naturalness is relevant -- our system should detect natural human turn-taking patterns, not the more rigid patterns of LLM-generated speech.

6. **Scale advantage**: At 1.5M conversations, SODA is large enough to train or fine-tune substantial models, unlike smaller dialogue corpora that have limited turn-taking pattern coverage.
