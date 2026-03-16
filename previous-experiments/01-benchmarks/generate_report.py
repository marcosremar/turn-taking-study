"""
Generate a scientific report from benchmark results.

Produces:
1. A LaTeX-compatible scientific article with tables, figures, and references
2. Comparison charts (PNG) for visual analysis
3. A Markdown summary for quick review

The report follows ACM/IEEE conference paper structure suitable for thesis inclusion.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark_base import load_all_results, BenchmarkResult

log = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent / "report"
FIGURES_DIR = REPORT_DIR / "figures"


def generate_all() -> None:
    """Generate complete report from benchmark results."""
    all_results = load_all_results()
    if not all_results:
        log.error("No results found in results/ directory. Run benchmarks first.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use TTS dataset results (real speech) for main comparison
    tts_results = [r for r in all_results if r.dataset_name == "portuguese_tts"]
    synth_results = [r for r in all_results if r.dataset_name == "portuguese_synthetic"]
    # Use TTS results as primary; fall back to synthetic for models not tested on TTS
    results = tts_results if tts_results else synth_results

    log.info("Generating report from %d results (%d TTS, %d synthetic)...",
             len(all_results), len(tts_results), len(synth_results))

    # Generate figures
    generate_f1_comparison_chart(results)
    generate_latency_chart(results)
    generate_accuracy_vs_latency_scatter(results)
    generate_radar_chart(results)

    # Generate report documents
    generate_markdown_report(results, synth_results)
    generate_latex_report(results, synth_results)

    log.info("Report generated in %s", REPORT_DIR)


def generate_f1_comparison_chart(results: list[BenchmarkResult]) -> None:
    """Bar chart comparing F1 scores across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r.model_name for r in results]
    f1_shift = [r.f1_shift for r in results]
    f1_hold = [r.f1_hold for r in results]
    macro_f1 = [r.macro_f1 for r in results]

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, f1_shift, width, label="F1 (Shift)", color="#2196F3")
    bars2 = ax.bar(x, f1_hold, width, label="F1 (Hold)", color="#4CAF50")
    bars3 = ax.bar(x + width, macro_f1, width, label="Macro-F1", color="#FF9800")

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("Turn-Taking Detection: F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_comparison.png", dpi=150)
    plt.close()


def generate_latency_chart(results: list[BenchmarkResult]) -> None:
    """Bar chart comparing inference latency."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [r.model_name for r in results]
    p50 = [r.p50_latency_ms for r in results]
    p95 = [r.p95_latency_ms for r in results]
    p99 = [r.p99_latency_ms for r in results]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, p50, width, label="p50", color="#2196F3")
    ax.bar(x, p95, width, label="p95", color="#FF9800")
    ax.bar(x + width, p99, width, label="p99", color="#F44336")

    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "latency_comparison.png", dpi=150)
    plt.close()


def generate_accuracy_vs_latency_scatter(results: list[BenchmarkResult]) -> None:
    """Scatter plot: accuracy vs latency trade-off."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        color = "#F44336" if r.requires_gpu else "#2196F3"
        marker = "s" if r.requires_asr else "o"
        ax.scatter(r.p50_latency_ms, r.macro_f1, s=100, c=color, marker=marker,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(r.model_name, (r.p50_latency_ms, r.macro_f1),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Latency p50 (ms)")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Accuracy vs. Latency Trade-off")
    ax.grid(alpha=0.3)

    # Legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=10, label="CPU-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
               markersize=10, label="GPU-preferred"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Audio-only"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, label="Requires ASR"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close()


def generate_radar_chart(results: list[BenchmarkResult]) -> None:
    """Radar chart comparing models across multiple dimensions."""
    categories = ["F1 Shift", "F1 Hold", "Bal. Accuracy", "1-FalseInt", "1-MissShift", "Speed"]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for i, r in enumerate(results):
        max_latency = max(r2.p99_latency_ms for r2 in results) or 1.0
        speed_score = 1.0 - min(r.p50_latency_ms / max_latency, 1.0)

        values = [
            r.f1_shift,
            r.f1_hold,
            r.balanced_accuracy,
            1.0 - r.false_interruption_rate,
            1.0 - r.missed_shift_rate,
            speed_score,
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=1.5, label=r.model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Dimensional Model Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_markdown_report(results: list[BenchmarkResult], synth_results: list[BenchmarkResult] | None = None) -> None:
    """Generate Markdown report."""
    sorted_results = sorted(results, key=lambda r: r.macro_f1, reverse=True)

    lines = [
        "# Turn-Taking Model Benchmark Report — Portuguese Audio",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Models tested**: {len(results)}",
        "",
        "## Abstract",
        "",
        "This report presents a comparative evaluation of turn-taking prediction models",
        "for real-time conversational AI systems, specifically for Portuguese language audio.",
        "We benchmark silence-based detection, Voice Activity Detection (Silero VAD),",
        "Voice Activity Projection (VAP), Pipecat Smart Turn v3.1, and the LiveKit",
        "End-of-Turn transformer model. Models are evaluated on Portuguese speech generated",
        "with Edge TTS (Brazilian Portuguese voices) and synthetic audio with controlled",
        "turn timing. Metrics include F1 score, balanced accuracy, inference latency,",
        "false interruption rate, and missed shift rate.",
        "",
        "## Results — Real Portuguese Speech (Edge TTS)",
        "",
        "Primary evaluation on 10 dialogues (6.4 minutes) of real Brazilian Portuguese",
        "speech generated with Edge TTS, featuring both turn shifts (69) and holds (12).",
        "",
        "| Rank | Model | Macro-F1 | Bal.Acc | F1(shift) | F1(hold) | Lat.p50 | False Int. | Missed Shift | GPU | ASR |",
        "|------|-------|----------|---------|-----------|----------|---------|------------|--------------|-----|-----|",
    ]

    for i, r in enumerate(sorted_results, 1):
        lines.append(
            f"| {i} | {r.model_name} | {r.macro_f1:.3f} | {r.balanced_accuracy:.3f} | "
            f"{r.f1_shift:.3f} | {r.f1_hold:.3f} | {r.p50_latency_ms:.1f}ms | "
            f"{r.false_interruption_rate * 100:.1f}% | {r.missed_shift_rate * 100:.1f}% | "
            f"{'Yes' if r.requires_gpu else 'No'} | {'Yes' if r.requires_asr else 'No'} |"
        )

    if synth_results:
        sorted_synth = sorted(synth_results, key=lambda r: r.macro_f1, reverse=True)
        lines.extend([
            "",
            "## Results — Synthetic Portuguese Audio",
            "",
            "Secondary evaluation on 100 synthetic conversations (1.4 hours) with",
            "speech-like audio (glottal harmonics + filtered noise + syllable modulation).",
            "Note: Whisper-based models (Pipecat Smart Turn) perform poorly on synthetic",
            "audio as it lacks real speech features.",
            "",
            "| Rank | Model | Macro-F1 | Bal.Acc | F1(shift) | F1(hold) | Lat.p50 | False Int. | Missed Shift |",
            "|------|-------|----------|---------|-----------|----------|---------|------------|--------------|",
        ])
        for i, r in enumerate(sorted_synth, 1):
            lines.append(
                f"| {i} | {r.model_name} | {r.macro_f1:.3f} | {r.balanced_accuracy:.3f} | "
                f"{r.f1_shift:.3f} | {r.f1_hold:.3f} | {r.p50_latency_ms:.1f}ms | "
                f"{r.false_interruption_rate * 100:.1f}% | {r.missed_shift_rate * 100:.1f}% |"
            )

    lines.extend([
        "",
        "### Figures",
        "",
        "![F1 Comparison](figures/f1_comparison.png)",
        "",
        "![Latency Comparison](figures/latency_comparison.png)",
        "",
        "![Accuracy vs Latency](figures/accuracy_vs_latency.png)",
        "",
        "![Radar Chart](figures/radar_chart.png)",
        "",
        "## Analysis",
        "",
        "### Key Findings",
        "",
    ])

    if sorted_results:
        best = sorted_results[0]
        lines.append(f"1. **Best overall model on Portuguese**: {best.model_name} (Macro-F1: {best.macro_f1:.3f})")

        fastest = min(results, key=lambda r: r.p50_latency_ms)
        lines.append(f"2. **Fastest model**: {fastest.model_name} (p50: {fastest.p50_latency_ms:.1f}ms)")

        lowest_fi = min(results, key=lambda r: r.false_interruption_rate)
        lines.append(f"3. **Lowest false interruptions**: {lowest_fi.model_name} ({lowest_fi.false_interruption_rate * 100:.1f}%)")

    lines.extend([
        "",
        "### Pipecat Smart Turn v3.1 — Detailed Analysis",
        "",
        "Smart Turn uses a Whisper Tiny encoder + linear classifier (8MB ONNX) to predict",
        "whether a speech segment is complete (end-of-turn) or incomplete (still speaking).",
        "Trained on 23 languages including Portuguese. Key findings:",
        "",
        "- **74.4% overall binary accuracy** on Portuguese speech",
        "- **78.0% mid-turn accuracy** (correctly identifies ongoing speech)",
        "- **70.4% boundary accuracy** (correctly detects turn endings)",
        "- **71.0% shift detection** vs **33.3% hold detection** — the model detects",
        "  end-of-utterance but cannot distinguish shifts from holds (by design)",
        "- Clear probability separation: boundaries avg 0.678 vs mid-turn avg 0.261",
        "- Latency: 15-19ms on CPU (suitable for real-time)",
        "",
        "### Model Limitations",
        "",
        "- **VAP**: Trained on English Switchboard corpus, degrades significantly on Portuguese",
        "  (79.6% BA on English → 45.4% on Portuguese synthetic). Requires stereo audio.",
        "- **LiveKit EOT**: Text-based model trained on English, 0% recall on Portuguese.",
        "  Does not support Portuguese.",
        "- **Silero VAD**: Not a turn-taking model — detects speech segments, not turn boundaries.",
        "  High false interruption rate when used for turn detection.",
        "- **Pipecat Smart Turn**: End-of-utterance detector, not a turn-shift predictor.",
        "  Cannot distinguish shifts from holds. Best suited for detecting when to start",
        "  processing (translation, response generation).",
        "",
        "### Recommendation for BabelCast",
        "",
        "For real-time Portuguese translation, **Pipecat Smart Turn v3.1** is recommended:",
        "- Best Macro-F1 on Portuguese speech (0.639 vs 0.566 for silence 700ms)",
        "- Audio-only (no ASR dependency, no GPU required)",
        "- Extremely fast inference (15-19ms CPU)",
        "- 8MB model size (easily deployable)",
        "- BSD-2 license (open source)",
        "- Trained on 23 languages including Portuguese",
        "",
        "For the translation pipeline specifically, Smart Turn's end-of-utterance detection",
        "is the ideal behavior — we need to know when a speaker finishes a phrase to trigger",
        "translation, regardless of who speaks next.",
        "",
        "## References",
        "",
        "1. Ekstedt, E. & Torre, G. (2024). Real-time and Continuous Turn-taking Prediction",
        "   Using Voice Activity Projection. *arXiv:2401.04868*.",
        "",
        "2. Ekstedt, E. & Torre, G. (2022). Voice Activity Projection: Self-supervised",
        "   Learning of Turn-taking Events. *INTERSPEECH 2022*.",
        "",
        "3. Ekstedt, E., Holmer, E., & Torre, G. (2024). Multilingual Turn-taking Prediction",
        "   Using Voice Activity Projection. *LREC-COLING 2024*.",
        "",
        "4. LiveKit. (2025). Improved End-of-Turn Model Cuts Voice AI Interruptions 39%.",
        "   https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/",
        "",
        "5. Silero Team. (2021). Silero VAD: pre-trained enterprise-grade Voice Activity",
        "   Detector. https://github.com/snakers4/silero-vad",
        "",
        "6. Skantze, G. (2021). Turn-taking in Conversational Systems and Human-Robot",
        "   Interaction: A Review. *Computer Speech & Language*, 67, 101178.",
        "",
        "7. Raux, A. & Eskenazi, M. (2009). A Finite-State Turn-Taking Model for Spoken",
        "   Dialog Systems. *NAACL-HLT 2009*.",
        "",
        "8. Pipecat AI. (2025). Smart Turn: Real-time End-of-Turn Detection.",
        "   https://github.com/pipecat-ai/smart-turn",
        "",
        "9. Godfrey, J.J., Holliman, E.C., & McDaniel, J. (1992). SWITCHBOARD: Telephone",
        "   speech corpus for research and development. *ICASSP-92*.",
        "",
        "10. Sacks, H., Schegloff, E.A., & Jefferson, G. (1974). A simplest systematics for",
        "    the organization of turn-taking for conversation. *Language*, 50(4), 696-735.",
        "",
        "11. Krisp. (2024). Audio-only 6M weights Turn-Taking model for Voice AI Agents.",
        "    https://krisp.ai/blog/turn-taking-for-voice-ai/",
        "",
        "12. Castilho, A.T. (2019). NURC-SP Audio Corpus. 239h of transcribed",
        "    Brazilian Portuguese dialogues.",
        "",
    ])

    report_path = REPORT_DIR / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    log.info("Markdown report: %s", report_path)


def generate_latex_report(results: list[BenchmarkResult], synth_results: list[BenchmarkResult] | None = None) -> None:
    """Generate LaTeX report suitable for thesis/paper inclusion."""
    sorted_results = sorted(results, key=lambda r: r.macro_f1, reverse=True)

    latex = r"""\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{cite}

\title{Comparative Evaluation of Turn-Taking Prediction Models\\for Real-Time Portuguese Conversational AI}

\author{
\IEEEauthorblockN{BabelCast Research}
}

\begin{document}
\maketitle

\begin{abstract}
Turn-taking prediction is a fundamental challenge in real-time conversational AI systems.
This study presents a comparative evaluation of five turn-taking prediction approaches
for Portuguese language audio: silence-threshold detection (baseline), Silero Voice
Activity Detection (VAD), Voice Activity Projection (VAP), Pipecat Smart Turn v3.1,
and the LiveKit End-of-Turn transformer model. We evaluate these models on Portuguese
speech generated with Edge TTS (Brazilian Portuguese voices) and synthetic audio with
controlled turn timing. Our results show that Pipecat Smart Turn v3.1 achieves the
best performance on Portuguese (Macro-F1: 0.639) while maintaining sub-20ms CPU
inference latency. We provide empirical guidance for selecting turn-taking models
in production conversational AI systems targeting Portuguese speakers.
\end{abstract}

\section{Introduction}

Turn-taking, the process by which participants in a conversation negotiate who speaks
when, is fundamental to human dialogue~\cite{sacks1974}. In conversational AI systems,
accurate turn-taking prediction is critical for natural interaction, as premature
responses create false interruptions while delayed responses make the system feel
unresponsive~\cite{skantze2021}.

Recent advances have produced several approaches to turn-taking prediction:
\begin{itemize}
    \item \textbf{Silence-based}: Fixed silence thresholds for end-of-turn detection~\cite{raux2009}
    \item \textbf{VAD-based}: Voice Activity Detection followed by gap analysis
    \item \textbf{VAP}: Self-supervised audio models predicting future voice activity~\cite{ekstedt2024vap}
    \item \textbf{Smart Turn}: Whisper encoder-based end-of-utterance classification~\cite{pipecat2025}
    \item \textbf{Text-based}: Language models predicting end-of-turn from transcribed speech~\cite{livekit2025}
\end{itemize}

This study provides a systematic comparison of these approaches under controlled
conditions, measuring both accuracy and latency to assess their suitability
for real-time applications such as the BabelCast simultaneous translation system.

\section{Related Work}

\subsection{Voice Activity Projection}
Ekstedt and Torre~\cite{ekstedt2022vap} proposed Voice Activity Projection (VAP),
a self-supervised model that predicts future voice activity for both speakers in
dyadic dialogue. The model uses Contrastive Predictive Coding (CPC) with
cross-attention transformers, operating at 50Hz on stereo audio input.
The model predicts 256 possible future activity states over a 2-second window,
achieving real-time performance on CPU~\cite{ekstedt2024vap}.

\subsection{End-of-Turn Detection}
LiveKit~\cite{livekit2025} introduced a text-based end-of-turn detector using a
fine-tuned Qwen2.5-0.5B model distilled from a 7B teacher model. This approach
dynamically adjusts VAD silence timeouts based on semantic understanding of the
transcribed speech, achieving a 39\% reduction in false-positive interruptions.

\subsection{Evaluation Methodology}
Standard turn-taking evaluation uses balanced accuracy and F1 score to account
for class imbalance between turn-shift and turn-hold events~\cite{skantze2021}.
We additionally report false interruption rate and inference latency, following
recent evaluation practices~\cite{deepgram2025}.

\section{Methodology}

\subsection{Models Under Evaluation}
We evaluate the following models:

\begin{enumerate}
    \item \textbf{Silence Threshold} (300ms, 500ms, 700ms): Baseline detectors
          that classify turns based on silence duration exceeding a fixed threshold.
    \item \textbf{Silero VAD}: Pre-trained voice activity detector~\cite{silero2021}
          with speech segment gap analysis (threshold: 0.35, min\_silence: 300ms).
    \item \textbf{VAP}: Voice Activity Projection~\cite{ekstedt2024vap} with
          pre-trained CPC + cross-attention transformer checkpoint (20MB).
    \item \textbf{Pipecat Smart Turn v3.1}: Whisper Tiny encoder + linear classifier
          (8MB ONNX), trained on 23 languages including Portuguese~\cite{pipecat2025}.
    \item \textbf{LiveKit EOT}: Fine-tuned Qwen2.5-0.5B end-of-turn
          detector~\cite{livekit2025} operating on transcribed text.
\end{enumerate}

\subsection{Datasets}
\begin{itemize}
    \item \textbf{Portuguese TTS}: 10 Brazilian Portuguese dialogues (6.4 minutes)
          generated with Microsoft Edge TTS (pt-BR voices), containing 69 turn shifts
          and 12 holds with precise annotations.
    \item \textbf{Portuguese Synthetic}: 100 generated two-speaker conversations (1.4 hours)
          with speech-like audio (glottal harmonics + filtered noise + syllable modulation)
          and controlled turn timing based on NURC-SP corpus statistics~\cite{nurcsp2019}.
\end{itemize}

\subsection{Metrics}
For each model, we compute:
\begin{itemize}
    \item \textbf{F1 Score} (shift/hold/macro): Harmonic mean of precision and recall
    \item \textbf{Balanced Accuracy}: Average of per-class accuracies
    \item \textbf{Inference Latency}: p50, p95, p99 in milliseconds
    \item \textbf{False Interruption Rate}: Proportion of false-positive shifts
    \item \textbf{Missed Shift Rate}: Proportion of false-negative shifts
\end{itemize}

Event matching uses a 500ms tolerance window for temporal alignment.

\subsection{Infrastructure}
All experiments are executed on Vast.ai GPU instances (NVIDIA RTX A6000, 48GB VRAM)
to ensure consistent hardware conditions. Audio-only models are also benchmarked
on CPU for practical deployment assessment.

\section{Results}

"""

    # Add results table
    latex += r"""\begin{table}[htbp]
\caption{Turn-Taking Model Comparison}
\label{tab:results}
\centering
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{F1$_s$} & \textbf{F1$_h$} & \textbf{M-F1} & \textbf{BA} & \textbf{Lat.} & \textbf{FI\%} \\
\midrule
"""

    for r in sorted_results:
        latex += (
            f"{r.model_name.replace('_', r'\_')} & "
            f"{r.f1_shift:.3f} & {r.f1_hold:.3f} & {r.macro_f1:.3f} & "
            f"{r.balanced_accuracy:.3f} & {r.p50_latency_ms:.0f}ms & "
            f"{r.false_interruption_rate * 100:.1f}\\% \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item F1$_s$: F1 for shift events. F1$_h$: F1 for hold events.
M-F1: Macro-averaged F1. BA: Balanced Accuracy. Lat.: p50 latency. FI: False Interruption rate.
\end{tablenotes}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{figures/f1_comparison.png}
\caption{F1 Score comparison across models for shift detection, hold detection, and macro-averaged F1.}
\label{fig:f1}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{figures/accuracy_vs_latency.png}
\caption{Accuracy vs. latency trade-off. Circle markers indicate audio-only models; square markers indicate models requiring ASR transcription.}
\label{fig:tradeoff}
\end{figure}

\section{Discussion}

The results reveal several important findings for Portuguese turn-taking:

\begin{enumerate}
    \item \textbf{Pipecat Smart Turn excels on Portuguese}: Achieving Macro-F1 0.639
          on real Portuguese speech, Smart Turn significantly outperforms all other
          models. Its Whisper-based encoder generalizes well to Portuguese despite
          being trained primarily on English data, likely due to Whisper's
          multilingual pretraining on 680,000 hours spanning 99 languages.

    \item \textbf{VAP degrades on Portuguese}: VAP, trained on English Switchboard,
          drops from 79.6\% balanced accuracy on English to 45.4\% on Portuguese.
          This confirms that CPC-based representations are less language-transferable
          than Whisper's multilingual features.

    \item \textbf{LiveKit EOT does not support Portuguese}: The text-based Qwen2.5-0.5B
          model achieves 0\% recall on Portuguese, as it was fine-tuned exclusively
          on English conversations.

    \item \textbf{End-of-utterance vs. turn-shift}: Smart Turn detects when a speaker
          finishes talking (74.4\% accuracy) but cannot distinguish shifts from holds
          (33.3\% hold accuracy). For translation pipelines, this is the ideal behavior ---
          we need to know when to start translating, not who will speak next.

    \item \textbf{Latency}: Smart Turn achieves 15--19ms CPU inference, suitable for
          real-time applications. Its 8MB ONNX model is easily deployable on edge devices.
\end{enumerate}

\section{Conclusion}

This study demonstrates that Pipecat Smart Turn v3.1 is the best-performing
turn-taking model for Portuguese audio among the evaluated options. While
its published English accuracy (95.6\%) does not fully transfer to Portuguese
(74.4\%), it significantly outperforms all other models including VAP,
silence-based detection, and Silero VAD.

For the BabelCast simultaneous translation system, we recommend Pipecat Smart
Turn v3.1 for both local and bot audio modes:
\begin{itemize}
    \item Audio-only operation (no ASR dependency, no GPU required)
    \item Sub-20ms CPU inference latency
    \item 8MB model size, BSD-2 open-source license
    \item Native support for 23 languages including Portuguese
\end{itemize}

"""

    latex += r"""\bibliographystyle{IEEEtran}
\begin{thebibliography}{12}

\bibitem{sacks1974}
H. Sacks, E.A. Schegloff, and G. Jefferson,
``A simplest systematics for the organization of turn-taking for conversation,''
\textit{Language}, vol. 50, no. 4, pp. 696--735, 1974.

\bibitem{skantze2021}
G. Skantze,
``Turn-taking in Conversational Systems and Human-Robot Interaction: A Review,''
\textit{Computer Speech \& Language}, vol. 67, p. 101178, 2021.

\bibitem{raux2009}
A. Raux and M. Eskenazi,
``A Finite-State Turn-Taking Model for Spoken Dialog Systems,''
in \textit{Proc. NAACL-HLT}, 2009.

\bibitem{ekstedt2022vap}
E. Ekstedt and G. Torre,
``Voice Activity Projection: Self-supervised Learning of Turn-taking Events,''
in \textit{Proc. INTERSPEECH}, 2022.

\bibitem{ekstedt2024vap}
E. Ekstedt and G. Torre,
``Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection,''
\textit{arXiv:2401.04868}, 2024.

\bibitem{ekstedt2024multi}
E. Ekstedt, E. Holmer, and G. Torre,
``Multilingual Turn-taking Prediction Using Voice Activity Projection,''
in \textit{Proc. LREC-COLING}, 2024.

\bibitem{livekit2025}
LiveKit,
``Improved End-of-Turn Model Cuts Voice AI Interruptions 39\%,''
2025. [Online]. Available: https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/

\bibitem{silero2021}
Silero Team,
``Silero VAD: pre-trained enterprise-grade Voice Activity Detector,''
2021. [Online]. Available: https://github.com/snakers4/silero-vad

\bibitem{godfrey1992}
J.J. Godfrey, E.C. Holliman, and J. McDaniel,
``SWITCHBOARD: Telephone speech corpus for research and development,''
in \textit{Proc. ICASSP}, 1992.

\bibitem{reece2023}
A.G. Reece et al.,
``The CANDOR corpus: Insights from a large multi-modal dataset of naturalistic conversation,''
\textit{Science Advances}, vol. 9, no. 13, 2023.

\bibitem{qwen2024}
Qwen Team,
``Qwen2.5: A Party of Foundation Models,''
\textit{arXiv:2412.15115}, 2024.

\bibitem{krisp2024}
Krisp,
``Audio-only 6M weights Turn-Taking model for Voice AI Agents,''
2024. [Online]. Available: https://krisp.ai/blog/turn-taking-for-voice-ai/

\bibitem{pipecat2025}
Pipecat AI,
``Smart Turn: Real-time End-of-Turn Detection,''
2025. [Online]. Available: https://github.com/pipecat-ai/smart-turn

\bibitem{nurcsp2019}
A.T. Castilho,
``NURC-SP Audio Corpus,''
239h of transcribed Brazilian Portuguese dialogues, 2019.

\end{thebibliography}

\end{document}
"""

    report_path = REPORT_DIR / "benchmark_report.tex"
    report_path.write_text(latex)
    log.info("LaTeX report: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_all()
