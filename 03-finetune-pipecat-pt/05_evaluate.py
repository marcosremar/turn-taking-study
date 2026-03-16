"""Evaluate fine-tuned model against Pipecat baseline and ONNX reference.

Comparisons:
1. Our fine-tuned model (PyTorch) on Pipecat PT test set
2. Pipecat original ONNX model on same test set
3. Threshold sweep to find optimal operating point
4. Per-source breakdown (pipecat data vs our TTS data)

Run:
    python 05_evaluate.py --model results/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
_workspace = Path("/workspace") if Path("/workspace").exists() else Path(".")
RESULTS_DIR = _workspace / "results"
CACHE_DIR = _workspace / "hf_cache"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE

# Pipecat baseline (from their blog: Smart Turn v3 Portuguese)
PIPECAT_BASELINE = {
    "accuracy": 0.9542,
    "fp_rate": 0.0279,
    "fn_rate": 0.0179,
    "source": "https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/",
}


def load_model(model_path: Path, device: str) -> nn.Module:
    """Load fine-tuned SmartTurnV3Model."""
    # Import from finetune script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "finetune", Path(__file__).parent / "04_finetune.py"
    )
    finetune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finetune)

    model = finetune.SmartTurnV3Model(whisper_model="openai/whisper-tiny")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    log.info("Loaded model from %s (epoch %d, val_f1=%.4f)",
             model_path, checkpoint.get("epoch", -1), checkpoint.get("val_f1", -1))
    return model


def load_onnx_model(onnx_dir: Path | None = None):
    """Load Pipecat ONNX model for comparison."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ONNX evaluation")
        return None

    if onnx_dir is None:
        onnx_dir = DATA_DIR / "pipecat_model"

    cpu_path = onnx_dir / "smart-turn-v3.2-cpu.onnx"
    gpu_path = onnx_dir / "smart-turn-v3.2-gpu.onnx"

    onnx_path = cpu_path if cpu_path.exists() else (gpu_path if gpu_path.exists() else None)
    if onnx_path is None:
        log.warning("No Pipecat ONNX model found in %s", onnx_dir)
        return None

    session = ort.InferenceSession(str(onnx_path))
    log.info("Loaded ONNX model: %s", onnx_path)
    return session


def load_test_data() -> tuple[list, list]:
    """Load test datasets: Pipecat PT test + our TTS test."""
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    # 1. Pipecat test data
    pipecat_samples = []
    test_dir = DATA_DIR / "pipecat_pt_test"
    if test_dir.exists():
        for wav_path in sorted(test_dir.glob("*.wav")):
            try:
                audio, sr = sf.read(str(wav_path))
                audio = np.array(audio, dtype=np.float32)
                if sr != SAMPLE_RATE:
                    import torchaudio
                    tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

                audio = _extract_window(audio)
                if audio is not None:
                    label = 1.0 if "_complete" in wav_path.name else 0.0
                    features = feature_extractor(
                        audio, sampling_rate=SAMPLE_RATE,
                        return_tensors="np", padding="max_length",
                        max_length=WINDOW_SAMPLES, truncation=True,
                        do_normalize=True,
                    ).input_features.squeeze(0).astype(np.float32)

                    pipecat_samples.append({
                        "features": features,
                        "label": label,
                        "source": "pipecat_test",
                        "file": wav_path.name,
                    })
            except Exception as e:
                pass

    log.info("Pipecat test: %d samples", len(pipecat_samples))

    # 2. Our TTS test data
    tts_samples = []
    tts_dir = DATA_DIR / "tts_dataset"
    meta_path = tts_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

        # Use last 20% as test
        test_start = int(len(metadata) * 0.8)
        test_meta = metadata[test_start:]

        audio_dir = tts_dir / "audio"
        for meta in test_meta:
            wav_path = audio_dir / meta["file"]
            if not wav_path.exists():
                continue
            try:
                audio, sr = sf.read(str(wav_path))
                audio = np.array(audio, dtype=np.float32)
                if sr != SAMPLE_RATE:
                    import torchaudio
                    tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

                audio = _extract_window(audio)
                if audio is not None:
                    label = 1.0 if meta["label"] == "complete" else 0.0
                    features = feature_extractor(
                        audio, sampling_rate=SAMPLE_RATE,
                        return_tensors="np", padding="max_length",
                        max_length=WINDOW_SAMPLES, truncation=True,
                        do_normalize=True,
                    ).input_features.squeeze(0).astype(np.float32)

                    tts_samples.append({
                        "features": features,
                        "label": label,
                        "source": meta.get("accent", "unknown"),
                        "file": meta["file"],
                    })
            except Exception:
                pass

    log.info("TTS test: %d samples", len(tts_samples))
    return pipecat_samples, tts_samples


def _extract_window(audio: np.ndarray) -> np.ndarray | None:
    """Extract 8s window from end of audio."""
    if len(audio) < SAMPLE_RATE:
        return None
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
    elif len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (WINDOW_SAMPLES - len(audio), 0), mode="constant")
    audio[-int(0.2 * SAMPLE_RATE):] = 0.0
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_pytorch(
    model: nn.Module,
    samples: list[dict],
    device: str,
    threshold: float = 0.5,
) -> dict:
    """Evaluate PyTorch model on samples."""
    if not samples:
        return {}

    tp = fp = fn = tn = 0
    latencies = []

    with torch.no_grad():
        for s in samples:
            features = torch.from_numpy(s["features"]).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            logits = model(features)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            pred = float(torch.sigmoid(logits).item() > threshold)
            label = s["label"]

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    return _compute_metrics(tp, fp, fn, tn, latencies)


def evaluate_onnx(
    session,
    samples: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Evaluate ONNX model on samples."""
    if not samples or session is None:
        return {}

    tp = fp = fn = tn = 0
    latencies = []

    input_name = session.get_inputs()[0].name
    for s in samples:
        features = s["features"][np.newaxis, ...]

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: features})
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        logit = outputs[0].item() if outputs[0].size == 1 else outputs[0][0].item()
        pred = float(1.0 / (1.0 + np.exp(-logit)) > threshold)
        label = s["label"]

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    return _compute_metrics(tp, fp, fn, tn, latencies)


def _compute_metrics(tp, fp, fn, tn, latencies=None) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fp_rate = fp / max(fp + tn, 1)
    fn_rate = fn / max(fn + tp, 1)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "fn_rate": round(fn_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }

    if latencies:
        metrics["latency_mean_ms"] = round(np.mean(latencies), 2)
        metrics["latency_p50_ms"] = round(np.median(latencies), 2)
        metrics["latency_p95_ms"] = round(np.percentile(latencies, 95), 2)

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model_path: Path) -> dict:
    """Run full evaluation suite."""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Evaluating on device: %s", device)

    # Load models
    model = load_model(model_path, device)
    onnx_session = load_onnx_model()

    # Load test data
    pipecat_samples, tts_samples = load_test_data()
    all_samples = pipecat_samples + tts_samples

    results = {"pipecat_baseline": PIPECAT_BASELINE}

    # 1. Our model on all test data
    log.info("\n=== Our model (all test data) ===")
    our_all = evaluate_pytorch(model, all_samples, device)
    if our_all:
        _log_metrics("Our model (all)", our_all)
        results["our_model_all"] = our_all

    # 2. Our model on Pipecat test only
    if pipecat_samples:
        log.info("\n=== Our model (Pipecat PT test only) ===")
        our_pipecat = evaluate_pytorch(model, pipecat_samples, device)
        _log_metrics("Our model (Pipecat test)", our_pipecat)
        results["our_model_pipecat_test"] = our_pipecat

        # Compare with baseline
        diff_acc = our_pipecat["accuracy"] - PIPECAT_BASELINE["accuracy"]
        diff_fp = our_pipecat["fp_rate"] - PIPECAT_BASELINE["fp_rate"]
        diff_fn = our_pipecat["fn_rate"] - PIPECAT_BASELINE["fn_rate"]
        log.info("  vs Pipecat baseline: accuracy %+.2f%%, FP %+.2f%%, FN %+.2f%%",
                 diff_acc * 100, diff_fp * 100, diff_fn * 100)

    # 3. Our model per accent
    for accent_name, accent_filter in [("native_pt_br", "native_pt_br"), ("french_pt", "french_pt")]:
        accent_samples = [s for s in tts_samples if s["source"] == accent_filter]
        if accent_samples:
            log.info("\n=== Our model (%s) ===", accent_name)
            m = evaluate_pytorch(model, accent_samples, device)
            _log_metrics(f"Our model ({accent_name})", m)
            results[f"our_model_{accent_name}"] = m

    # 4. ONNX reference model
    if onnx_session and pipecat_samples:
        log.info("\n=== Pipecat ONNX model (reference) ===")
        onnx_metrics = evaluate_onnx(onnx_session, pipecat_samples)
        if onnx_metrics:
            _log_metrics("Pipecat ONNX", onnx_metrics)
            results["pipecat_onnx"] = onnx_metrics

    # 5. Threshold sweep
    log.info("\n=== Threshold Sweep (all test data) ===")
    sweep = {}
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        m = evaluate_pytorch(model, all_samples, device, threshold=thresh)
        sweep[str(thresh)] = m
        log.info("  threshold=%.2f: acc=%.3f prec=%.3f rec=%.3f f1=%.3f FP=%.3f FN=%.3f",
                 thresh, m["accuracy"], m["precision"], m["recall"],
                 m["f1"], m["fp_rate"], m["fn_rate"])

    results["threshold_sweep"] = sweep

    # Find optimal thresholds
    best_f1_thresh = max(sweep.items(), key=lambda x: x[1]["f1"])
    best_prec_thresh = max(
        [(k, v) for k, v in sweep.items() if v["recall"] >= 0.90],
        key=lambda x: x[1]["precision"],
        default=(None, None),
    )

    log.info("\n=== Recommendations ===")
    log.info("  Best F1: threshold=%.2f (F1=%.3f, prec=%.3f, rec=%.3f)",
             float(best_f1_thresh[0]), best_f1_thresh[1]["f1"],
             best_f1_thresh[1]["precision"], best_f1_thresh[1]["recall"])
    if best_prec_thresh[0]:
        log.info("  Best precision (recall>=90%%): threshold=%.2f (prec=%.3f, rec=%.3f)",
                 float(best_prec_thresh[0]), best_prec_thresh[1]["precision"],
                 best_prec_thresh[1]["recall"])

    # ----- Dual Threshold for Language Learning Avatar -----
    # Inspired by Deepgram Flux (eot_threshold + eager_eot_threshold)
    # For L2 learners: higher final threshold to avoid interrupting mid-hesitation
    log.info("\n=== Dual Threshold (Language Learning Mode) ===")
    log.info("  Context: conversational avatar for francophones learning Portuguese")
    log.info("  Priority: minimize interruptions (FP) over fast response (FN)")

    # Final threshold: minimize FP rate while keeping recall >= 85%
    final_candidates = [(k, v) for k, v in sweep.items()
                        if v["recall"] >= 0.85 and float(k) >= 0.6]
    if final_candidates:
        best_final = min(final_candidates, key=lambda x: x[1]["fp_rate"])
        final_thresh = float(best_final[0])
        log.info("  Final threshold: %.2f (FP=%.1f%%, rec=%.1f%%)",
                 final_thresh, best_final[1]["fp_rate"] * 100,
                 best_final[1]["recall"] * 100)
    else:
        final_thresh = 0.7
        log.info("  Final threshold: 0.70 (default)")

    # Eager threshold: lower confidence for speculative LLM generation
    eager_thresh = max(0.3, final_thresh - 0.3)
    log.info("  Eager threshold: %.2f (start speculative LLM prep)", eager_thresh)
    log.info("  Expected latency savings: ~150-250ms on response start")

    # Simulate dual-threshold behavior
    if all_samples:
        log.info("\n  Dual-threshold simulation:")
        eager_m = evaluate_pytorch(model, all_samples, device, threshold=eager_thresh)
        final_m = evaluate_pytorch(model, all_samples, device, threshold=final_thresh)
        log.info("    Eager (%.2f): would trigger on %.1f%% of samples (%.1f%% false triggers)",
                 eager_thresh, (eager_m["tp"] + eager_m["fp"]) / max(eager_m["total"], 1) * 100,
                 eager_m["fp_rate"] * 100)
        log.info("    Final (%.2f): confirms %.1f%% of turns (%.1f%% false confirms, %.1f%% missed)",
                 final_thresh, final_m["recall"] * 100,
                 final_m["fp_rate"] * 100, final_m["fn_rate"] * 100)
        wasted_speculative = eager_m["fp"] - final_m["fp"]
        log.info("    Wasted speculative preps: %d (started but not confirmed)", max(0, wasted_speculative))

    results["recommended"] = {
        "best_f1_threshold": float(best_f1_thresh[0]),
        "best_precision_threshold": float(best_prec_thresh[0]) if best_prec_thresh[0] else None,
        "dual_threshold": {
            "eager": eager_thresh,
            "final": final_thresh,
            "mode": "language_learning",
            "rationale": "Higher final threshold minimizes interruptions for L2 learners "
                        "who pause 500-2000ms mid-sentence. Eager threshold enables "
                        "speculative LLM prep for lower perceived latency.",
        },
    }

    # 6. Summary comparison table
    log.info("\n" + "=" * 70)
    log.info("SUMMARY COMPARISON")
    log.info("=" * 70)
    log.info("%-30s %8s %8s %8s %8s", "Model", "Acc", "FP%", "FN%", "F1")
    log.info("-" * 70)
    log.info("%-30s %7.1f%% %7.2f%% %7.2f%% %8s",
             "Pipecat v3.2 (baseline)",
             PIPECAT_BASELINE["accuracy"] * 100,
             PIPECAT_BASELINE["fp_rate"] * 100,
             PIPECAT_BASELINE["fn_rate"] * 100,
             "~0.96")

    if "pipecat_onnx" in results:
        m = results["pipecat_onnx"]
        log.info("%-30s %7.1f%% %7.2f%% %7.2f%% %7.3f",
                 "Pipecat ONNX (our eval)",
                 m["accuracy"] * 100, m["fp_rate"] * 100,
                 m["fn_rate"] * 100, m["f1"])

    if "our_model_pipecat_test" in results:
        m = results["our_model_pipecat_test"]
        log.info("%-30s %7.1f%% %7.2f%% %7.2f%% %7.3f",
                 "Our model (Pipecat test)",
                 m["accuracy"] * 100, m["fp_rate"] * 100,
                 m["fn_rate"] * 100, m["f1"])

    if "our_model_all" in results:
        m = results["our_model_all"]
        log.info("%-30s %7.1f%% %7.2f%% %7.2f%% %7.3f",
                 "Our model (all test)",
                 m["accuracy"] * 100, m["fp_rate"] * 100,
                 m["fn_rate"] * 100, m["f1"])

    log.info("=" * 70)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved to %s", results_path)

    return results


def _log_metrics(name: str, m: dict) -> None:
    log.info("%s:", name)
    log.info("  Accuracy:  %.3f (%.1f%%)", m["accuracy"], m["accuracy"] * 100)
    log.info("  Precision: %.3f", m["precision"])
    log.info("  Recall:    %.3f", m["recall"])
    log.info("  F1:        %.3f", m["f1"])
    log.info("  FP rate:   %.3f (%.1f%%)", m["fp_rate"], m["fp_rate"] * 100)
    log.info("  FN rate:   %.3f (%.1f%%)", m["fn_rate"], m["fn_rate"] * 100)
    log.info("  TP=%d FP=%d FN=%d TN=%d (total=%d)", m["tp"], m["fp"], m["fn"], m["tn"], m["total"])
    if "latency_mean_ms" in m:
        log.info("  Latency: mean=%.1fms p50=%.1fms p95=%.1fms",
                 m["latency_mean_ms"], m["latency_p50_ms"], m["latency_p95_ms"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(RESULTS_DIR / "best_model.pt"),
                        help="Path to fine-tuned model checkpoint")
    args = parser.parse_args()

    run_evaluation(Path(args.model))
