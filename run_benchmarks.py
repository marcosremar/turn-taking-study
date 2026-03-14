"""
Run all turn-taking benchmarks and generate results.

Usage:
    python run_benchmarks.py --all                    # Run everything
    python run_benchmarks.py --models vap silero_vad  # Run specific models
    python run_benchmarks.py --dataset synthetic      # Use specific dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmark_base import evaluate_model, save_result, load_all_results, BenchmarkResult
from setup_dataset import generate_synthetic_dataset, download_switchboard_from_hf, load_annotations

log = logging.getLogger(__name__)


def get_model(name: str):
    """Factory for turn-taking models."""
    if name == "silence_500ms":
        from benchmark_silence import SilenceThresholdModel
        return SilenceThresholdModel(silence_threshold_ms=500.0)
    elif name == "silence_700ms":
        from benchmark_silence import SilenceThresholdModel
        return SilenceThresholdModel(silence_threshold_ms=700.0)
    elif name == "silence_1000ms":
        from benchmark_silence import SilenceThresholdModel
        return SilenceThresholdModel(silence_threshold_ms=1000.0)
    elif name == "silero_vad":
        from benchmark_silero_vad import SileroVADModel
        return SileroVADModel()
    elif name == "vap":
        from benchmark_vap import VAPModel
        return VAPModel()
    elif name == "livekit_eot":
        from benchmark_livekit_eot import LiveKitEOTModel
        return LiveKitEOTModel()
    else:
        raise ValueError(f"Unknown model: {name}")


ALL_MODELS = [
    "silence_500ms",
    "silence_700ms",
    "silence_1000ms",
    "silero_vad",
    "vap",
    "livekit_eot",
]


def run_benchmarks(
    model_names: list[str],
    dataset_name: str = "synthetic",
    n_synthetic: int = 100,
    tolerance_ms: float = 500.0,
) -> list[BenchmarkResult]:
    """Run benchmarks for specified models on a dataset."""

    # Prepare dataset
    log.info("=== Preparing dataset: %s ===", dataset_name)
    try:
        conversations = load_annotations(dataset_name)
        log.info("Loaded %d cached annotations", len(conversations))
    except FileNotFoundError:
        if dataset_name == "synthetic":
            conversations = generate_synthetic_dataset(n_conversations=n_synthetic)
        elif dataset_name == "switchboard":
            conversations = download_switchboard_from_hf()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if not conversations:
        log.error("No conversations loaded!")
        return []

    log.info("Dataset: %d conversations, %.1f hours",
             len(conversations), sum(c.duration for c in conversations) / 3600)

    # Run each model
    results: list[BenchmarkResult] = []
    for model_name in model_names:
        log.info("=== Benchmarking: %s ===", model_name)
        try:
            model = get_model(model_name)
            result = evaluate_model(model, conversations, dataset_name, tolerance_ms)
            save_result(result)
            results.append(result)

            log.info(
                "  F1(shift)=%.3f  F1(hold)=%.3f  Balanced-Acc=%.3f  "
                "Latency(p50)=%.1fms  FalseInterrupt=%.2f%%",
                result.f1_shift, result.f1_hold, result.balanced_accuracy,
                result.p50_latency_ms, result.false_interruption_rate * 100,
            )
        except Exception as e:
            log.error("Failed to benchmark %s: %s", model_name, e, exc_info=True)

    return results


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a comparison table of all results."""
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback
        for r in results:
            print(f"{r.model_name}: F1={r.macro_f1:.3f} BalAcc={r.balanced_accuracy:.3f} "
                  f"Latency={r.p50_latency_ms:.1f}ms")
        return

    headers = [
        "Model", "F1(shift)", "F1(hold)", "Macro-F1", "Bal.Acc",
        "Latency(p50)", "FalseInt%", "GPU?", "ASR?", "Size(MB)"
    ]
    rows = []
    for r in sorted(results, key=lambda x: x.macro_f1, reverse=True):
        rows.append([
            r.model_name,
            f"{r.f1_shift:.3f}",
            f"{r.f1_hold:.3f}",
            f"{r.macro_f1:.3f}",
            f"{r.balanced_accuracy:.3f}",
            f"{r.p50_latency_ms:.1f}ms",
            f"{r.false_interruption_rate * 100:.1f}%",
            "Yes" if r.requires_gpu else "No",
            "Yes" if r.requires_asr else "No",
            f"{r.model_size_mb:.0f}",
        ])

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run turn-taking benchmarks")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to benchmark (default: all). Options: {ALL_MODELS}")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "switchboard"],
                        help="Dataset to use (default: synthetic)")
    parser.add_argument("--n-synthetic", type=int, default=100,
                        help="Number of synthetic conversations (default: 100)")
    parser.add_argument("--tolerance-ms", type=float, default=500.0,
                        help="Event matching tolerance in ms (default: 500)")
    parser.add_argument("--all", action="store_true",
                        help="Run all models on all datasets")
    args = parser.parse_args()

    model_names = args.models or ALL_MODELS

    if args.all:
        # Run on all datasets
        all_results = []
        for ds in ["synthetic", "switchboard"]:
            try:
                res = run_benchmarks(model_names, ds, args.n_synthetic, args.tolerance_ms)
                all_results.extend(res)
            except Exception as e:
                log.error("Failed on dataset %s: %s", ds, e)
        print_comparison_table(all_results)
    else:
        results = run_benchmarks(model_names, args.dataset, args.n_synthetic, args.tolerance_ms)
        print_comparison_table(results)
