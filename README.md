# Turn-Taking Model Benchmark Study

Comparative evaluation of turn-taking prediction models for real-time conversational AI.

## Models Evaluated

| Model | Type | Size | GPU Required | ASR Required |
|-------|------|------|--------------|--------------|
| Silence Threshold (500/700/1000ms) | Rule-based | 0 | No | No |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Audio DNN | 2MB | No | No |
| [VAP](https://github.com/ErikEkstedt/VoiceActivityProjection) | Audio Transformer | 20MB | Optional | No |
| [LiveKit EOT](https://huggingface.co/livekit/turn-detector) | Text Transformer | 281MB | No | Yes |

## Quick Start

### Local (CPU)

```bash
pip install -r requirements.txt
python setup_dataset.py --dataset synthetic
python run_benchmarks.py --models silence_700ms silero_vad --dataset synthetic
python generate_report.py
```

### Vast.ai (GPU)

```bash
# Build and deploy
export VAST_API_KEY="your_key"
python deploy_vast.py --all

# Or step by step:
python deploy_vast.py --build
python deploy_vast.py --deploy --gpu "RTX A6000"
python deploy_vast.py --collect --instance-id <ID>
python deploy_vast.py --cleanup --instance-id <ID>
```

### Docker

```bash
docker build -t turn-taking-study .
docker run --gpus all turn-taking-study python run_benchmarks.py --all
```

## Datasets

- **Synthetic**: Auto-generated two-speaker conversations with exact ground truth
- **Switchboard**: Natural telephone conversations ([HuggingFace](https://huggingface.co/datasets/hhoangphuoc/switchboard))

## Metrics

- **F1 Score** (shift/hold/macro) — precision-recall balance
- **Balanced Accuracy** — handles class imbalance
- **Inference Latency** (p50/p95/p99) — real-time feasibility
- **False Interruption Rate** — premature turn predictions
- **Missed Shift Rate** — failed turn-change detection

## Output

After running benchmarks:
- `results/` — JSON files with per-model metrics
- `report/benchmark_report.md` — Markdown summary with tables and figures
- `report/benchmark_report.tex` — LaTeX article (IEEE format) for thesis inclusion
- `report/figures/` — PNG charts (F1 comparison, latency, radar, scatter)

## Project Structure

```
turn-taking-study/
├── Dockerfile                  # GPU-ready container
├── requirements.txt            # Python dependencies
├── setup_dataset.py            # Dataset download & preparation
├── benchmark_base.py           # Base classes & evaluation metrics
├── benchmark_silence.py        # Silence threshold baseline
├── benchmark_silero_vad.py     # Silero VAD model
├── benchmark_vap.py            # Voice Activity Projection model
├── benchmark_livekit_eot.py    # LiveKit End-of-Turn model
├── run_benchmarks.py           # Orchestrator script
├── deploy_vast.py              # Vast.ai deployment automation
├── generate_report.py          # Report & figure generation
└── report/                     # Generated output
    ├── benchmark_report.md
    ├── benchmark_report.tex
    └── figures/
```

## References

1. Ekstedt, E. & Torre, G. (2024). *Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection*. arXiv:2401.04868.
2. Ekstedt, E. & Torre, G. (2022). *Voice Activity Projection: Self-supervised Learning of Turn-taking Events*. INTERSPEECH 2022.
3. LiveKit. (2025). *Improved End-of-Turn Model Cuts Voice AI Interruptions 39%*.
4. Silero Team. (2021). *Silero VAD: pre-trained enterprise-grade Voice Activity Detector*.
5. Skantze, G. (2021). *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review*. Computer Speech & Language, 67.
6. Sacks, H., Schegloff, E.A., & Jefferson, G. (1974). *A simplest systematics for the organization of turn-taking for conversation*. Language, 50(4).
7. Raux, A. & Eskenazi, M. (2009). *A Finite-State Turn-Taking Model for Spoken Dialog Systems*. NAACL-HLT.
8. Godfrey, J.J., et al. (1992). *SWITCHBOARD: Telephone speech corpus for research and development*. ICASSP-92.
9. Reece, A.G., et al. (2023). *The CANDOR corpus*. Science Advances, 9(13).
10. Qwen Team. (2024). *Qwen2.5: A Party of Foundation Models*. arXiv:2412.15115.
11. Krisp. (2024). *Audio-only 6M weights Turn-Taking model for Voice AI Agents*.

## License

MIT
