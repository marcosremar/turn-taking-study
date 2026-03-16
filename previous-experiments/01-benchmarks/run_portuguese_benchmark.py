"""
Run turn-taking benchmarks on Portuguese audio data.

Tests all models on synthetic Portuguese conversations to evaluate
which performs best for Portuguese language turn-taking detection.

Usage:
    python run_portuguese_benchmark.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from setup_portuguese_dataset import (
    generate_portuguese_synthetic,
    load_annotations,
    Conversation,
    TurnSegment,
)
from benchmark_base import (
    evaluate_model,
    save_result,
    BenchmarkResult,
    TurnTakingModel,
    PredictedEvent,
    RESULTS_DIR,
)


def run_all_benchmarks():
    """Run all benchmarks on Portuguese data."""

    # Step 1: Generate dataset
    log.info("=" * 60)
    log.info("STEP 1: Generating Portuguese synthetic dataset")
    log.info("=" * 60)

    try:
        conversations = load_annotations("portuguese_synthetic")
        log.info("Loaded cached annotations: %d conversations", len(conversations))
    except FileNotFoundError:
        conversations = generate_portuguese_synthetic(n_conversations=100)

    total_hours = sum(c.duration for c in conversations) / 3600
    total_shifts = sum(len(c.turn_shifts) for c in conversations)
    total_holds = sum(len(c.holds) for c in conversations)
    log.info("Dataset: %d conversations, %.1f hours, %d shifts, %d holds",
             len(conversations), total_hours, total_shifts, total_holds)

    # Step 2: Run each model
    results: list[BenchmarkResult] = []
    models_to_test = []

    # 2a: Silence baselines
    log.info("=" * 60)
    log.info("STEP 2a: Silence threshold baselines")
    log.info("=" * 60)

    from benchmark_silence import SilenceThresholdModel
    for threshold_ms in [300, 500, 700, 1000]:
        models_to_test.append(SilenceThresholdModel(silence_threshold_ms=threshold_ms))

    # 2b: Silero VAD
    log.info("=" * 60)
    log.info("STEP 2b: Silero VAD")
    log.info("=" * 60)

    from benchmark_silero_vad import SileroVADModel
    models_to_test.append(SileroVADModel())

    # 2c: VAP
    log.info("=" * 60)
    log.info("STEP 2c: Voice Activity Projection (VAP)")
    log.info("=" * 60)

    try:
        from benchmark_vap import VAPModel
        models_to_test.append(VAPModel())
    except Exception as e:
        log.error("VAP not available: %s", e)

    # 2d: LiveKit EOT
    log.info("=" * 60)
    log.info("STEP 2d: LiveKit End-of-Turn")
    log.info("=" * 60)

    try:
        from benchmark_livekit_eot import LiveKitEOTModel
        models_to_test.append(LiveKitEOTModel())
    except Exception as e:
        log.error("LiveKit EOT not available: %s", e)

    # Run all models
    for model in models_to_test:
        log.info("-" * 40)
        log.info("Running: %s", model.name)
        log.info("-" * 40)

        t0 = time.time()
        try:
            result = evaluate_model(
                model, conversations, "portuguese_synthetic", tolerance_ms=500.0
            )
            elapsed = time.time() - t0
            save_result(result)
            results.append(result)

            log.info("  RESULTS for %s:", model.name)
            log.info("    F1(shift)      = %.4f", result.f1_shift)
            log.info("    F1(hold)       = %.4f", result.f1_hold)
            log.info("    Macro-F1       = %.4f", result.macro_f1)
            log.info("    Balanced Acc   = %.4f", result.balanced_accuracy)
            log.info("    Precision(s)   = %.4f", result.precision_shift)
            log.info("    Recall(s)      = %.4f", result.recall_shift)
            log.info("    Latency p50    = %.1f ms", result.p50_latency_ms)
            log.info("    Latency p95    = %.1f ms", result.p95_latency_ms)
            log.info("    FalseInterrupt = %.2f%%", result.false_interruption_rate * 100)
            log.info("    MissedShift    = %.2f%%", result.missed_shift_rate * 100)
            log.info("    Time elapsed   = %.1f s", elapsed)
        except Exception as e:
            log.error("  FAILED: %s", e, exc_info=True)

    # Step 3: Print comparison
    log.info("=" * 60)
    log.info("STEP 3: FINAL COMPARISON")
    log.info("=" * 60)
    print_comparison(results)

    # Step 4: Generate report
    log.info("=" * 60)
    log.info("STEP 4: Generating report")
    log.info("=" * 60)
    try:
        from generate_report import generate_all
        generate_all()
        log.info("Report generated in report/")
    except Exception as e:
        log.error("Report generation failed: %s", e)

    return results


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print final comparison table."""
    if not results:
        log.warning("No results to compare!")
        return

    try:
        from tabulate import tabulate
    except ImportError:
        for r in sorted(results, key=lambda x: x.macro_f1, reverse=True):
            print(f"  {r.model_name:20s}  F1={r.macro_f1:.3f}  BA={r.balanced_accuracy:.3f}  "
                  f"Lat={r.p50_latency_ms:.0f}ms  FI={r.false_interruption_rate*100:.1f}%")
        return

    headers = [
        "Rank", "Model", "Macro-F1", "Bal.Acc",
        "F1(shift)", "F1(hold)", "Prec(s)", "Rec(s)",
        "Lat.p50", "Lat.p95", "FalseInt%", "MissShift%",
        "GPU?", "ASR?", "Size(MB)"
    ]

    sorted_results = sorted(results, key=lambda r: r.macro_f1, reverse=True)
    rows = []
    for i, r in enumerate(sorted_results, 1):
        rows.append([
            i,
            r.model_name,
            f"{r.macro_f1:.4f}",
            f"{r.balanced_accuracy:.4f}",
            f"{r.f1_shift:.4f}",
            f"{r.f1_hold:.4f}",
            f"{r.precision_shift:.4f}",
            f"{r.recall_shift:.4f}",
            f"{r.p50_latency_ms:.1f}ms",
            f"{r.p95_latency_ms:.1f}ms",
            f"{r.false_interruption_rate * 100:.1f}%",
            f"{r.missed_shift_rate * 100:.1f}%",
            "Yes" if r.requires_gpu else "No",
            "Yes" if r.requires_asr else "No",
            f"{r.model_size_mb:.0f}",
        ])

    print("\n" + "=" * 120)
    print("TURN-TAKING BENCHMARK — PORTUGUESE AUDIO")
    print("=" * 120)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()

    # Winner
    best = sorted_results[0]
    print(f"WINNER: {best.model_name}")
    print(f"  Macro-F1: {best.macro_f1:.4f}")
    print(f"  Balanced Accuracy: {best.balanced_accuracy:.4f}")
    print(f"  False Interruption Rate: {best.false_interruption_rate*100:.1f}%")
    print(f"  Latency (p50): {best.p50_latency_ms:.1f}ms")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("benchmark_portuguese.log"),
        ],
    )
    run_all_benchmarks()
