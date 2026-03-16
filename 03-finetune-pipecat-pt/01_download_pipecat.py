"""Download Pipecat Smart Turn v3 dataset + model architecture.

The Pipecat model is only published as ONNX (no PyTorch weights).
So we initialize from openai/whisper-tiny and train using their dataset
which already includes Portuguese samples.

This script:
1. Downloads Pipecat's training dataset (270K samples, 23 langs)
2. Filters Portuguese samples
3. Downloads the ONNX model for reference/benchmarking
4. Saves Portuguese subset locally
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path("/workspace/hf_cache") if Path("/workspace").exists() else DATA_DIR / "hf_cache"


def download_pipecat_dataset(max_pt_samples: int = 5000) -> Path:
    """Download Pipecat v3.2 training data and extract Portuguese samples."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download full dataset (streaming to avoid 37GB download)
    log.info("Loading Pipecat Smart Turn v3.2 training data (streaming)...")
    ds = load_dataset(
        "pipecat-ai/smart-turn-data-v3.2-train",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )

    # Filter Portuguese samples
    pt_samples = []
    other_count = 0
    for row in ds:
        lang = row.get("language", "")
        if lang == "por":
            pt_samples.append({
                "id": row["id"],
                "language": lang,
                "endpoint_bool": row["endpoint_bool"],
                "midfiller": row.get("midfiller", False),
                "endfiller": row.get("endfiller", False),
                "synthetic": row.get("synthetic", True),
                "dataset": row.get("dataset", ""),
                "spoken_text": row.get("spoken_text", ""),
                "audio": row["audio"],
            })
            if len(pt_samples) % 100 == 0:
                log.info("  Found %d Portuguese samples (scanned %d others)",
                         len(pt_samples), other_count)
            if len(pt_samples) >= max_pt_samples:
                break
        else:
            other_count += 1

    log.info("Total: %d Portuguese samples found (scanned %d other languages)",
             len(pt_samples), other_count)

    # Save metadata (without audio)
    meta_path = DATA_DIR / "pipecat_pt_metadata.json"
    meta = [{k: v for k, v in s.items() if k != "audio"} for s in pt_samples]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Metadata saved to %s", meta_path)

    # Save audio files
    audio_dir = DATA_DIR / "pipecat_pt_audio"
    audio_dir.mkdir(exist_ok=True)

    import soundfile as sf
    import numpy as np

    complete = 0
    incomplete = 0
    for s in pt_samples:
        audio_data = s["audio"]
        audio = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        label = "complete" if s["endpoint_bool"] else "incomplete"
        if s["endpoint_bool"]:
            complete += 1
        else:
            incomplete += 1

        out_path = audio_dir / f"{s['id']}_{label}.wav"
        sf.write(str(out_path), audio, sr)

    log.info("Audio saved: %d complete, %d incomplete → %s",
             complete, incomplete, audio_dir)

    return audio_dir


def download_pipecat_test_data(max_pt_samples: int = 2000) -> Path:
    """Download Pipecat test data for evaluation."""
    from datasets import load_dataset

    log.info("Loading Pipecat Smart Turn v3.2 test data (streaming)...")
    ds = load_dataset(
        "pipecat-ai/smart-turn-data-v3.2-test",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )

    pt_samples = []
    for row in ds:
        if row.get("language", "") == "por":
            pt_samples.append(row)
            if len(pt_samples) >= max_pt_samples:
                break

    log.info("Found %d Portuguese test samples", len(pt_samples))

    # Save
    test_dir = DATA_DIR / "pipecat_pt_test"
    test_dir.mkdir(exist_ok=True)

    import soundfile as sf
    import numpy as np

    for s in pt_samples:
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        label = "complete" if s["endpoint_bool"] else "incomplete"
        sf.write(str(test_dir / f"{s['id']}_{label}.wav"), audio, sr)

    log.info("Test audio saved → %s", test_dir)
    return test_dir


def download_onnx_model() -> Path:
    """Download Pipecat ONNX model for benchmarking."""
    from huggingface_hub import hf_hub_download

    model_dir = DATA_DIR / "pipecat_model"
    model_dir.mkdir(exist_ok=True)

    for fname in ["smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-gpu.onnx"]:
        path = hf_hub_download(
            "pipecat-ai/smart-turn-v3",
            fname,
            local_dir=str(model_dir),
        )
        log.info("Downloaded %s → %s", fname, path)

    return model_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    log.info("=== Step 1: Download Pipecat Portuguese training data ===")
    download_pipecat_dataset(max_pt_samples=5000)

    log.info("\n=== Step 2: Download Pipecat Portuguese test data ===")
    download_pipecat_test_data(max_pt_samples=2000)

    log.info("\n=== Step 3: Download ONNX model for benchmarking ===")
    download_onnx_model()

    log.info("\nDone!")
