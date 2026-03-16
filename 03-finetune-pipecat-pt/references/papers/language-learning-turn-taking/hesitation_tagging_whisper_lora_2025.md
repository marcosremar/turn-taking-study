# Acoustically Precise Hesitation Tagging Is Essential for End-to-End Verbatim Transcription Systems

**PDF:** `hesitation_tagging_l2_whisper_lora.pdf`
**Source:** https://arxiv.org/pdf/2506.04076
**Authors:** Jhen-Ke Lin, Hao-Chien Lu, Chung-Chun Wang, Hong-Yun Lin, Berlin Chen (National Taiwan Normal University)
**Venue:** arXiv 2025 (Speak & Improve Challenge 2025 submission)

## Summary

Demonstrates that acoustically precise hesitation tagging (labeling "um" and "uh" accurately) significantly improves ASR for L2 English learners when fine-tuning Whisper with LoRA. Compares three transcription schemes and shows that explicit filled-pause labeling yields an 11.3% relative WER improvement over omitting hesitations.

## Method

### Three Transcription Schemes Compared
1. **Pure**: All hesitation tags and punctuation removed (baseline)
2. **Rich**: Generic "#" tags for hesitations + punctuation markers (., ?, ...)
3. **Extra**: Acoustically precise "um"/"uh" tokens inferred by Gemini 2.0 Flash from audio-transcript pairs + punctuation

### Gemini-based Annotation
- Used Google Gemini 2.0 Flash as an offline labeling tool
- Input: Rich transcription + corresponding audio
- Task: Infer acoustically plausible filled pauses from the generic "#" markers
- Cost: **Only $5 USD** to label entire training + dev set (thousands of utterances)
- Dramatically cheaper than human annotation

### Fine-tuning Approach
- **Whisper Large V3** (1.55B params) for challenge tracks
- **Whisper Large V3 Turbo** (809M params, distilled) for post-challenge experiments
- **rsLoRA** (rank-stabilized Low-Rank Adaptation): rank=32, alpha=8, dropout=0.05
- Applied to query, key, value, output projections + feedforward layers
- Trained on Speak & Improve Corpus 2025 (~55h transcribed subset)

## Key Results

### Baseline (no fine-tuning)
| Model | Parameters | WER |
|-------|-----------|-----|
| Whisper Small | 244M | 10.7% |
| Whisper Medium | 769M | 10.4% |
| Whisper Large V3 | 1.55B | 9.5% |
| Whisper Large V3 Turbo | 809M | 9.6% |

### Challenge Results
- **Closed Track** (Pure scheme): 6.47% WER -- **1st place**
- **Open Track** (Extra scheme): 5.81% WER -- 3rd place

### Post-Challenge: Transcription Scheme Comparison (Whisper Large V3 Turbo)
| Scheme | WER | vs. Pure |
|--------|-----|----------|
| Pure | 6.2% | baseline |
| Rich | 7.2% | +16.1% (worse!) |
| **Extra** | **5.5%** | **-11.3%** |

### Critical Finding
- Generic "#" tags actually **hurt** performance (+16.1% relative WER increase)
- Acoustically precise "um"/"uh" **significantly help** (-11.3% relative WER improvement)
- Abstract/generic tags conflate different types of non-lexical vocalizations
- Real filled-pause tokens strengthen alignment between acoustic patterns and transcript output

## Relevance to Turn-Taking for Language Learners

- **Hesitations are turn-taking signals**: In L2 speech, filled pauses ("um", "uh") often signal that the speaker is still thinking and has NOT yielded the turn. Accurate detection of these pauses is critical for turn-taking systems.
- **Practical LoRA approach**: The LoRA fine-tuning approach (rank=32 on Whisper) is directly applicable to fine-tuning ASR for Portuguese L2 learners with minimal compute.
- **Cheap annotation via LLM**: Using Gemini/Claude to annotate hesitations at $5/dataset makes it feasible to create L2 Portuguese hesitation-annotated data.
- **Verbatim transcription enables turn-taking**: If the ASR system accurately captures filled pauses, the downstream turn-taking model gets richer signal about speaker state (thinking vs. yielding).
- **L2-specific challenges**: L2 speakers produce more and different hesitation patterns than L1 speakers. Models trained only on L1 speech will mishandle these.
- **Whisper as foundation**: Confirms Whisper Large V3 as a strong foundation for L2 speech, reducible from 9.5% to 5.5% WER with targeted fine-tuning on only ~55 hours of data.
