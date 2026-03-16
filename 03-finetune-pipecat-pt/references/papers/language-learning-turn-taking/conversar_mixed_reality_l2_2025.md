# ConversAR: Mixed Reality Agents for Interactive Group Conversation (L2 Learning)

**PDF:** `conversar_mixed_reality_l2.pdf`
**Source:** https://arxiv.org/pdf/2510.08227
**Authors:** Mariana Fernandez-Espinosa, Kai Zhang, Jad Bendarkawi, et al. (University of Notre Dame, Princeton, NJIT)
**Venue:** arXiv 2025 (CHI-style HCI paper)

## Summary

Presents ConversAR, a Mixed Reality system using Generative AI and XR (Meta Quest 3) to support situated, personalized group conversations for L2 language learners. Features embodied AI agents, scene recognition, and generative 3D props anchored to real-world surroundings. Tested with 21 L2 learners.

## System Architecture

### Key Components
- **Platform**: Unity on Meta Quest 3
- **LLM Backend**: OpenAI API (GPT-5) -- not fine-tuned
- **NPCs**: Two AI agents (Ready Player Me avatars + Mixamo animations)
- **Speech**: Speech-to-text + Text-to-speech pipeline
- **Scene Recognition**: Meta Passthrough Camera API -> GPT-5 for object recognition
- **3D Props**: Text-to-3D generative AI for contextual objects during conversation

### Interaction Flow
1. **Getting to Know You Phase** (1-on-1): Single NPC assesses learner's level, interests, strengths/weaknesses
2. **Scene Capture**: System photographs physical surroundings, identifies objects
3. **Multi-party Conversation**: Two NPCs engage learner in group conversation grounded in their environment
4. **Dynamic Props**: 3D objects generated based on conversation topics, placed on physical surfaces

### Turn-Taking Implementation
- **Supervisor LLM** manages turn-taking to ensure balanced engagement
- NPCs converse with each other for up to 3 consecutive turns (pilot-tested limit)
- NPCs consistently address the learner with direct questions or invitations
- Learner gets infinite thinking time (no timer pressure)
- Turn assignment is dynamic, based on natural dialogue flow (e.g., if learner mentions an NPC by name)

## Formative Study Findings (10 SLA Educators)

### Key Insights About L2 Learners in Group Conversations
1. **KI1**: Students fear mistakes and judgment when speaking
2. **KI2**: Students fear being left out when peers have higher proficiency (more proficient speakers dominate)
3. **KI3**: Students fear repeating mistakes because corrective feedback is missing (leads to "fossilization of errors")
4. **KI4**: Students disengage when activities feel irrelevant to their lives

## Design Goals
1. **DG1**: Facilitate confidence through group conversations with AI peers (no social pressure)
2. **DG2**: Deliver supportive corrective feedback (recasts, clarification requests, metalinguistic feedback)
3. **DG3**: Create realistic, contextualized conversations grounded in physical environment
4. **DG4**: Interactive 3D props to sustain and deepen conversation
5. **DG5**: Adaptive scaffolding matching learner proficiency level (vocabulary, sentence complexity)

## User Study Results (21 L2 Learners)
- System enhanced learner engagement
- Increased willingness to communicate
- Offered a safe space for speaking practice
- Generative 3D props helped sustain and deepen conversations
- Corrective feedback was well-received (implicit recasts preferred)

## Corrective Feedback Strategies
- **Recasts**: NPC repeats learner's utterance correctly without explicitly pointing out error
- **Clarification requests**: Signal misunderstanding, prompting reformulation
- **Metalinguistic feedback**: Comments about correctness without giving the answer
- **Circumlocution**: Rephrasing concepts in simpler terms for low-proficiency learners

## Relevance to Turn-Taking for Language Learners

- **Group dynamics matter**: L2 learners face unique turn-taking challenges in group settings -- fear of speaking, proficiency gaps, dominance by stronger speakers. A turn-taking system must account for these.
- **Infinite thinking time**: The system deliberately removes time pressure, acknowledging that L2 speakers need more time to formulate responses. This contrasts sharply with standard end-of-turn detectors that use fixed silence thresholds.
- **Supervisor LLM for turn management**: Using an LLM to manage turn-taking (rather than acoustic signals alone) is a practical approach for L2 group conversations where acoustic cues may be unreliable.
- **Balanced participation**: The system ensures NPCs invite the learner to contribute, preventing the common classroom problem of silent learners.
- **Corrective feedback integrated into turn-taking**: The NPCs use their turn to provide implicit corrections (recasts), which is a pedagogically sound approach that doesn't disrupt conversational flow.
- **Situated/contextual conversations**: Grounding conversations in the physical environment provides natural topic anchors, reducing the cognitive load of thinking of what to say (a major source of long pauses in L2 speech).
- **Proficiency-adaptive language**: The system adjusts complexity based on assessed level, which is relevant for calibrating turn-taking expectations (lower proficiency -> expect longer pauses).
