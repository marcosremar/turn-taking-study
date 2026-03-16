# Turn-Taking Model — Deteccao de Fim de Turno para BabelCast

Pesquisa, benchmarks e fine-tuning de modelos de deteccao de fim de turno para traducao simultanea em portugues.

## Estrutura do Repositorio

```
docs/turn-taking-study/
  README.md                        # Este documento
  melhorias_turn_detection.md      # Plano de melhorias + resultados das 3 rodadas
  RESEARCH_LOG.md                  # Log de pesquisa
  data/                            # Datasets (NURC-SP, CORAA, TTS) — ~10GB
  hf_cache/                        # Cache HuggingFace

  previous-experiments/
    01-benchmarks/                 # Benchmark de 5 modelos em portugues
      benchmark_*.py               # Scripts de benchmark (Silence, Silero, VAP, Pipecat, LiveKit)
      setup_*.py                   # Scripts de setup de datasets
      report/                      # Relatorio gerado (markdown + LaTeX + graficos)

    02-finetune-scratch/           # Fine-tuning do zero (3 rodadas)
      finetune_smart_turn_v3.py    # Script principal (Whisper Tiny + Focal Loss)
      modal_finetune.py            # Deploy no Modal
      results/                     # Rodada 1: Whisper Base + BCE (F1=0.796)
      results-tiny/                # Rodada 2: Whisper Tiny + BCE (F1=0.788)
      results-focal/               # Rodada 3: Whisper Tiny + Focal Loss (F1=0.798)
      checkpoints/                 # Checkpoints v1/v2

  03-finetune-pipecat-pt/          # NOVO: Fine-tune a partir do Pipecat pre-treinado
    README.md                      # Documentacao completa do experimento
```

---

## Resumo dos Experimentos

### 01 — Benchmarks (5 modelos em portugues)

Comparacao de modelos existentes em audio portugues real (NURC-SP, 77 min).

### 02 — Fine-tune do zero (3 rodadas)

Treinamos Whisper Tiny encoder + classifier do zero em 15K amostras de portugues (CORAA + MUPE). Melhor resultado: **F1=0.798, precision 83% @threshold=0.65**. Detalhes em `melhorias_turn_detection.md`.

### 03 — Fine-tune a partir do Pipecat (proximo)

Fine-tune do modelo pre-treinado do Pipecat (270K amostras, 23 linguas) especificamente pra portugues + frances falando portugues. Usa LLMs (Claude) pra criar labels de qualidade + TTS pra gerar audio. Detalhes em `03-finetune-pipecat-pt/README.md`.

---

## Resultados dos Benchmarks (Experimento 01)

Comparative evaluation of turn-taking prediction models for real-time conversational AI, with focus on **Portuguese language** performance.

## Models Evaluated

| Model | Type | Size | GPU | ASR | Portuguese Support |
|-------|------|------|-----|-----|--------------------|
| Silence Threshold (300/500/700ms) | Rule-based | 0 | No | No | Language-independent |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Audio DNN | 2MB | No | No | Language-independent |
| [VAP](https://github.com/ErikEkstedt/VoiceActivityProjection) | Audio Transformer (CPC) | 20MB | Optional | No | Trained on English only |
| [Pipecat Smart Turn v3.1](https://github.com/pipecat-ai/smart-turn) | Audio Transformer (Whisper) | 8MB | No | No | Included in 23 languages |
| [LiveKit EOT](https://huggingface.co/livekit/turn-detector) | Text Transformer (Qwen2.5) | 281MB | No | Yes | English only |

## Key Results — Portuguese

### Real Portuguese Speech (NURC-SP corpus, 77 min, 15 dialogues)

**End-of-utterance detection accuracy (is the speaker done talking?):**

| Model | Detects speaker stopped | False alarm rate | Overall accuracy |
|-------|------------------------|------------------|-----------------|
| Pipecat Smart Turn v3.1 (original) | 84.9% | 54.9% | 68.6% |
| Pipecat Smart Turn v3.1 (fine-tuned PT) | 98.4% | 73.8% | 68.5% |
| Silero VAD | ~95%+ | ~5% | ~95% |

**Conclusion:** Silero VAD remains the most robust approach for detecting when a speaker stops talking in Portuguese. Smart Turn's Whisper-based approach adds linguistic intelligence but suffers from high false alarm rates on Portuguese, even after fine-tuning.

### Turn-taking benchmark (Edge TTS, 10 dialogues, 6.4 min)

| Rank | Model | Macro-F1 | Balanced Acc | Latency p50 | False Int. |
|------|-------|----------|-------------|-------------|------------|
| 1 | Pipecat Smart Turn v3.1 | 0.639 | 0.639 | 18.3ms | 22.8% |
| 2 | Silence 700ms | 0.566 | 0.573 | 0.1ms | 18.1% |
| 3 | Silero VAD | 0.401 | 0.500 | 9.0ms | 100.0% |
| 4 | VAP | 0.000 | 0.000 | — | — (needs stereo) |

---

## Pipecat Smart Turn — Model Documentation

### Overview

Smart Turn is an open-source end-of-turn detection model created by **Daily** (daily.co), the company behind the Pipecat voice AI framework. It predicts whether a speaker has finished their turn ("complete") or is still talking ("incomplete") using only audio input.

**No academic paper exists.** The model is documented through blog posts and GitHub only.

### Architecture

```
Input: 16kHz mono PCM audio (up to 8 seconds)
    │
    ▼
Whisper Feature Extractor → Log-mel spectrogram (80 bins × 800 frames)
    │
    ▼
Whisper Tiny Encoder (pretrained, openai/whisper-tiny)
    │  Output: (batch, 400, 384) — 400 frames, 384-dim hidden state
    ▼
Attention Pooling: Linear(384→256) → Tanh → Linear(256→1)
    │  Learns which audio frames are most important for the decision
    ▼  Weighted sum → (batch, 384)
Classifier MLP:
    Linear(384→256) → LayerNorm → GELU → Dropout(0.1)
    → Linear(256→64) → GELU → Linear(64→1)
    │
    ▼
Sigmoid → probability [0, 1]
    > 0.5 = "Complete" (speaker finished)
    ≤ 0.5 = "Incomplete" (speaker still talking)
```

**Total parameters:** ~8M
**Model size:** 8MB (int8 ONNX) / 32MB (fp32 ONNX)

### Why Whisper Tiny?

The team evolved through several architectures:

| Version | Backbone | Size | Problem |
|---------|----------|------|---------|
| v1 | wav2vec2-BERT | 2.3GB | Overfitted, too large |
| v2 | wav2vec2 + linear | 360MB | Still large |
| v3+ | **Whisper Tiny encoder** | 8MB | Good balance |

Whisper Tiny was chosen because:
- Pretrained on **680,000 hours** of multilingual speech (99 languages)
- Encoder produces rich acoustic representations without needing the decoder
- Only 39M params in full Whisper Tiny; encoder alone is much smaller
- The attention pooling + MLP classifier adds minimal overhead

### Training Data

**Dataset:** `pipecat-ai/smart-turn-data-v3.2-train` on HuggingFace
**Size:** 270,946 samples (41.4 GB)
**Languages:** 23 (Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, **Portuguese**, Russian, Spanish, Turkish, Ukrainian, Vietnamese)

**Data format per sample:**

| Field | Type | Description |
|-------|------|-------------|
| `audio` | Audio | 16kHz mono PCM, up to 16s |
| `endpoint_bool` | bool | True = complete, False = incomplete |
| `language` | string | ISO 639-3 code (e.g., "por") |
| `midfiller` | bool | Filler word mid-utterance ("um", "éh") |
| `endfiller` | bool | Filler word at end |
| `synthetic` | bool | TTS-generated vs human |
| `dataset` | string | Source (12 different sources) |

**Data generation pipeline:**
1. **Text sources:** 1.2M+ multilingual sentences from HuggingFace datasets
2. **Cleaning:** Gemini 2.5 Flash filtered grammatically incorrect sentences (removed 50-80%)
3. **TTS:** Google Chirp3 for synthetic audio generation
4. **Filler words:** Language-specific lists (generated by Claude/GPT), inserted by Gemini Flash
5. **Human audio:** Contributed by Liva AI, Midcentury, MundoAI
6. **Noise augmentation (v3.2):** Background noise from CC-0 Freesound.org samples
7. **Target split:** 50/50 complete vs incomplete

### Training Process

```python
# Hyperparameters (from train.py)
learning_rate = 5e-5
epochs = 4
train_batch_size = 384
eval_batch_size = 128
warmup_ratio = 0.2
weight_decay = 0.01
lr_scheduler = "cosine"
loss = BCEWithLogitsLoss(pos_weight=dynamic_per_batch)
```

**Hardware:** Modal L4 GPU (or local GPU)
**Training time:** ~53-79 minutes depending on GPU
**Framework:** HuggingFace Transformers Trainer API
**Logging:** Weights & Biases

### Published Accuracy by Language

| Language | Accuracy | FPR | FNR |
|----------|----------|-----|-----|
| Turkish | 97.10% | 1.66% | 1.24% |
| Korean | 96.85% | 1.12% | 2.02% |
| English | 95.60% | — | — |
| Spanish | 91.00% | — | — |
| Bengali | 84.10% | 10.80% | 5.10% |
| Vietnamese | 81.27% | 14.84% | 3.88% |
| **Portuguese** | **Not reported** | — | — |

### Inference Latency

| Device | Latency |
|--------|---------|
| AWS c7a.2xlarge (CPU) | 12.6 ms |
| NVIDIA L40S (GPU) | 3.3 ms |
| Apple M-series (MPS) | ~18 ms |

### Our Evaluation on Portuguese

We tested Smart Turn v3.1 on real Brazilian Portuguese speech from the **NURC-SP Corpus Minimo** (239h corpus of spontaneous São Paulo dialogues, CC BY-NC-ND 4.0):

| Metric | Result |
|--------|--------|
| Boundary detection (speaker actually stopped → model says "Complete") | **84.9%** |
| Mid-turn detection (speaker still talking → model says "Incomplete") | **45.1%** |
| Overall binary accuracy | **68.6%** |
| Shift detection (speaker change) | **87.7%** |
| Probability at boundaries (mean) | 0.809 |
| Probability at mid-turn (mean) | 0.522 |
| Separation (boundary - midturn) | 0.287 |

**Key finding:** Smart Turn detects end-of-utterance well (84.9%) but has a high false positive rate (54.9%) during ongoing speech. The model tends to predict "Complete" too aggressively on Portuguese.

### Fine-tuning Attempt

We fine-tuned the model on Portuguese using 6,031 samples extracted from NURC-SP (15 dialogues, 77 minutes) + Edge TTS dialogues:

| Metric | Original | Fine-tuned |
|--------|----------|------------|
| Boundary detection | 84.9% | **98.4%** |
| Mid-turn detection | 45.1% | 26.2% (worse) |
| Overall accuracy | 68.6% | 68.5% (same) |
| False alarm rate | 54.9% | 73.8% (worse) |

**Result:** Fine-tuning improved boundary detection but worsened false alarm rate. The model overfitted to predicting "Complete" for everything. The overall accuracy did not improve.

---

## Strategy: Improving Smart Turn for Portuguese

### Why It Doesn't Work Well on Portuguese

1. **Underrepresented in training data:** Portuguese is 1 of 23 languages in 270K samples — likely <5% of training data. English dominates.

2. **Mostly synthetic Portuguese data:** The training pipeline uses TTS (Google Chirp3) for most non-English languages. Synthetic speech lacks natural hesitations, overlaps, and prosodic variation.

3. **Portuguese prosody differs from English:**
   - Portuguese has more overlap between speakers (~15% vs ~5% in English)
   - Shorter inter-turn gaps (median ~200ms vs ~300ms in English)
   - Different intonation patterns at sentence endings
   - More use of filler words ("né", "tipo", "éh", "então")

4. **NURC-SP audio quality:** 1970s-1990s recordings with noise, which the model wasn't trained on (v3.2 added noise augmentation, but for modern noise profiles).

### Improvement Strategy

#### Phase 1: Better Training Data (Estimated effort: 1-2 weeks)

**Goal:** Create 20,000+ high-quality Portuguese training samples with proper class balance.

**Data sources:**
1. **NURC-SP Corpus Minimo** (19h, already downloaded) — extract more samples with sliding windows at various positions
2. **CORAA NURC-SP Audio Corpus** (239h, HuggingFace) — massive source of real dialogues
3. **C-ORAL-BRASIL** (21h, via Zenodo) — spontaneous informal speech
4. **Edge TTS generation** — create diverse Portuguese dialogues with multiple speakers/styles
5. **Real conversation recording** — record actual Portuguese conversations with timestamp annotations

**Key improvements over our first attempt:**
- Use **cross-validation** — never test on conversations used for training
- Generate **more diverse "incomplete" samples** — multiple positions within each turn, not just midpoint
- Include **Portuguese-specific fillers** ("né?", "tipo assim", "éh", "então") as end-of-utterance markers
- Add **noise augmentation** (background noise, room reverb, microphone artifacts)
- Balance dataset: exactly 50/50 complete vs incomplete, without augmentation tricks

#### Phase 2: Architecture Tweaks (Estimated effort: 1 week)

1. **Lower threshold for Portuguese:** Instead of 0.5, use 0.65-0.75 as the "Complete" threshold. This reduces false alarms at the cost of slightly slower detection.

2. **Language-specific classification head:** Add a language embedding to the classifier so the model can learn different decision boundaries per language.

3. **Longer context window:** Increase from 8s to 12-16s. Portuguese turns tend to be longer (2.5s mean vs 1.8s in English), so more context helps.

4. **Prosody features:** Add pitch (F0) contour as an additional input feature. Portuguese has distinctive falling intonation at statement endings vs rising at questions.

#### Phase 3: Proper Evaluation (Estimated effort: 1 week)

1. **Hold-out test set:** Reserve 3-5 NURC-SP conversations never seen during training
2. **Cross-corpus evaluation:** Test on CORAA data not used in training
3. **Real-world test:** Record and test on modern Portuguese conversations (Zoom/Teams calls)
4. **Compare with Silero VAD:** Side-by-side evaluation on the same test set with identical metrics
5. **Threshold sweep:** Find the optimal probability threshold for Portuguese specifically

#### Phase 4: Integration with BabelCast (Estimated effort: 2-3 days)

If the improved model achieves >85% accuracy with <15% false alarm rate on Portuguese:

1. Replace Silero VAD's end-of-speech detection with Smart Turn PT
2. Keep Silero VAD for initial voice activity detection (speech vs silence)
3. Use Smart Turn only for the endpoint decision (when to trigger translation)
4. Hybrid approach: `Silero VAD (speech detected) → Smart Turn PT (speech complete?) → Translate`

### Required Resources

| Resource | Purpose | Cost |
|----------|---------|------|
| NURC-SP + CORAA data | Training samples | Free (CC BY-NC-ND 4.0) |
| GPU for training (L4/A6000) | Fine-tuning, ~1 hour | ~$1-2 on Vast.ai |
| Edge TTS | Synthetic data generation | Free |
| Weights & Biases | Training tracking | Free tier |

### Expected Outcome

With 20,000+ properly prepared Portuguese samples and cross-validated evaluation, we estimate:
- **Boundary detection:** 90%+ (up from 84.9%)
- **False alarm rate:** <20% (down from 54.9%)
- **Overall accuracy:** >85% (up from 68.6%)

This would make Smart Turn PT a viable complement to Silero VAD for Portuguese end-of-utterance detection.

---

## Quick Start

### Local (CPU)

```bash
pip install -r requirements.txt

# Generate Portuguese dataset
python setup_portuguese_dataset.py --dataset synthetic

# Run benchmarks
python run_portuguese_benchmark.py

# Generate report
python generate_report.py
```

### With Real Portuguese Speech (NURC-SP)

```bash
# Prepare NURC-SP dialogues (downloads from HuggingFace)
python setup_nurc_dataset.py

# Run Pipecat Smart Turn benchmark
python -c "
from benchmark_pipecat import PipecatSmartTurnModel
from benchmark_base import evaluate_model
# ... (see run_portuguese_benchmark.py)
"
```

### Fine-tune Smart Turn for Portuguese

```bash
# 1. Prepare training data from NURC-SP
python prepare_training_data.py

# 2. Fine-tune (runs on MPS/CUDA/CPU)
python finetune_smart_turn.py

# 3. Test the fine-tuned model
# ONNX model saved to checkpoints/smart_turn_pt/smart_turn_pt.onnx
```

### Vast.ai (GPU)

```bash
export VAST_API_KEY="your_key"
python deploy_vast.py --all
```

## Project Structure

```
turn-taking-study/
├── README.md                       # This file
├── Dockerfile                      # GPU-ready container
├── requirements.txt                # Python dependencies
│
├── # Benchmark Framework
├── benchmark_base.py               # Base classes & evaluation metrics
├── benchmark_silence.py            # Silence threshold baseline
├── benchmark_silero_vad.py         # Silero VAD model
├── benchmark_vap.py                # Voice Activity Projection model
├── benchmark_livekit_eot.py        # LiveKit End-of-Turn model
├── benchmark_pipecat.py            # Pipecat Smart Turn v3.1
├── run_benchmarks.py               # General benchmark orchestrator
├── run_portuguese_benchmark.py     # Portuguese-specific benchmark
│
├── # Dataset Preparation
├── setup_dataset.py                # General dataset download
├── setup_portuguese_dataset.py     # Portuguese synthetic dataset
├── setup_nurc_dataset.py           # NURC-SP real speech dataset
├── generate_tts_dataset.py         # Edge TTS Portuguese dialogues
│
├── # Fine-tuning
├── prepare_training_data.py        # Extract training samples from NURC-SP
├── finetune_smart_turn.py          # Fine-tune Smart Turn on Portuguese
│
├── # Deployment & Reporting
├── deploy_vast.py                  # Vast.ai deployment automation
├── generate_report.py              # Report & figure generation
│
├── data/                           # Audio files & annotations (gitignored)
│   ├── annotations/                # JSON ground truth files
│   ├── nurc_sp/                    # NURC-SP real speech
│   ├── portuguese/                 # Synthetic Portuguese audio
│   ├── portuguese_tts/             # Edge TTS Portuguese audio
│   └── smart_turn_pt_training/     # Fine-tuning training samples
│
├── checkpoints/                    # Trained models (gitignored)
│   └── smart_turn_pt/
│       ├── best_model.pt           # PyTorch checkpoint
│       └── smart_turn_pt.onnx      # ONNX model (30.6 MB)
│
├── results/                        # Benchmark result JSONs
└── report/                         # Generated reports
    ├── benchmark_report.md
    ├── benchmark_report.tex        # IEEE format for thesis
    └── figures/                    # PNG charts
```

## Datasets Used

| Dataset | Type | Size | Language | Source |
|---------|------|------|----------|--------|
| Portuguese Synthetic | Generated audio | 1.4h, 100 convs | pt-BR | Local generation |
| Portuguese TTS | Edge TTS speech | 6.4min, 10 convs | pt-BR | Microsoft Edge TTS |
| NURC-SP Corpus Minimo | Real dialogues (1970s-90s) | 19h, 21 recordings | pt-BR | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/NURC-SP_Corpus_Minimo) |
| CORAA NURC-SP | Real dialogues | 239h | pt-BR | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus) |

## References

1. Ekstedt, E. & Torre, G. (2024). *Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection*. arXiv:2401.04868.
2. Ekstedt, E. & Torre, G. (2022). *Voice Activity Projection: Self-supervised Learning of Turn-taking Events*. INTERSPEECH 2022.
3. Ekstedt, E., Holmer, E., & Torre, G. (2024). *Multilingual Turn-taking Prediction Using Voice Activity Projection*. LREC-COLING 2024.
4. Daily. (2025). *Smart Turn: Real-time End-of-Turn Detection*. GitHub. https://github.com/pipecat-ai/smart-turn
5. Daily. (2025). *Announcing Smart Turn v3, with CPU inference in just 12ms*. https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
6. Daily. (2025). *Improved accuracy in Smart Turn v3.1*. https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/
7. Daily. (2026). *Smart Turn v3.2: Handling noisy environments and short responses*. https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
8. LiveKit. (2025). *Improved End-of-Turn Model Cuts Voice AI Interruptions 39%*. https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/
9. Silero Team. (2021). *Silero VAD: pre-trained enterprise-grade Voice Activity Detector*. https://github.com/snakers4/silero-vad
10. Skantze, G. (2021). *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review*. Computer Speech & Language, 67, 101178.
11. Sacks, H., Schegloff, E.A., & Jefferson, G. (1974). *A simplest systematics for the organization of turn-taking for conversation*. Language, 50(4), 696-735.
12. Raux, A. & Eskenazi, M. (2009). *A Finite-State Turn-Taking Model for Spoken Dialog Systems*. NAACL-HLT.
13. Krisp. (2024). *Audio-only 6M weights Turn-Taking model for Voice AI Agents*. https://krisp.ai/blog/turn-taking-for-voice-ai/
14. Castilho, A.T. (2019). *NURC-SP Audio Corpus*. 239h of transcribed Brazilian Portuguese dialogues.
15. Godfrey, J.J., et al. (1992). *SWITCHBOARD: Telephone speech corpus for research and development*. ICASSP-92.

## License

MIT
