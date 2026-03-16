"""
Prepare Portuguese training data for Smart Turn fine-tuning.

Takes NURC-SP real conversation segments and creates labeled samples:
- "complete": 8s window ending at a turn boundary (speaker finished)
- "incomplete": 8s window from mid-turn (speaker still talking)

Output: FLAC files organized in the directory structure expected by
smart-turn's raw_to_hf_dataset.py
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

TARGET_SR = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * TARGET_SR

OUTPUT_DIR = Path(__file__).parent / "data" / "smart_turn_pt_training" / "por"


def prepare_from_nurc(annotations_path: str, min_samples: int = 2000) -> dict:
    """Create training samples from NURC-SP annotations."""
    with open(annotations_path) as f:
        data = json.load(f)

    stats = {"complete": 0, "incomplete": 0, "skipped": 0}

    for conv_data in data:
        audio, sr = sf.read(conv_data["audio_path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        turns = conv_data["turns"]
        conv_id = conv_data["conv_id"]

        for i in range(len(turns)):
            turn = turns[i]
            turn_start = turn["start"]
            turn_end = turn["end"]
            turn_dur = turn_end - turn_start

            # --- COMPLETE samples: window ending at turn boundary ---
            if i > 0 and turn_dur > 0.5:
                boundary_t = turn_start
                end_sample = int(boundary_t * sr)
                start_sample = max(0, end_sample - WINDOW_SAMPLES)
                window = audio[start_sample:end_sample]

                if len(window) >= sr:  # At least 1s of audio
                    _save_sample(window, sr, "complete", "nofiller", conv_id, i)
                    stats["complete"] += 1

            # --- INCOMPLETE samples: windows during the turn ---
            if turn_dur >= 2.0:
                # Sample at multiple points within the turn
                n_points = max(1, int(turn_dur / 1.5))  # Every ~1.5s
                for p in range(n_points):
                    # Position within the turn (avoid the very end)
                    frac = (p + 0.5) / (n_points + 1)
                    if frac > 0.85:  # Don't sample too close to end
                        continue

                    mid_t = turn_start + turn_dur * frac
                    mid_sample = int(mid_t * sr)
                    start_sample = max(0, mid_sample - WINDOW_SAMPLES)
                    window = audio[start_sample:mid_sample]

                    if len(window) >= sr:
                        _save_sample(window, sr, "incomplete", "nofiller", conv_id, i, p)
                        stats["incomplete"] += 1

            # Also create a complete sample at the END of the last turn
            if i == len(turns) - 1 and turn_dur > 1.0:
                end_sample = min(int(turn_end * sr), len(audio))
                start_sample = max(0, end_sample - WINDOW_SAMPLES)
                window = audio[start_sample:end_sample]
                if len(window) >= sr:
                    _save_sample(window, sr, "complete", "nofiller", conv_id, i, 99)
                    stats["complete"] += 1

    return stats


def _save_sample(
    audio: np.ndarray,
    sr: int,
    endpoint: str,  # "complete" or "incomplete"
    filler: str,  # "nofiller", "midfiller", "endfiller"
    conv_id: str,
    turn_idx: int,
    sub_idx: int = 0,
) -> None:
    """Save a training sample as FLAC."""
    # Pad/truncate to exactly 8 seconds
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
    elif len(audio) < WINDOW_SAMPLES:
        padding = WINDOW_SAMPLES - len(audio)
        audio = np.pad(audio, (padding, 0), mode="constant", constant_values=0)

    # Add ~200ms silence at end (matching VAD behavior)
    silence = int(0.2 * sr)
    audio[-silence:] = 0.0

    out_dir = OUTPUT_DIR / f"{endpoint}-{filler}"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{conv_id}_t{turn_idx:03d}_s{sub_idx:02d}_{uuid.uuid4().hex[:8]}.flac"
    sf.write(str(out_dir / filename), audio, sr, format="FLAC", subtype="PCM_16")


def prepare_from_tts(annotations_path: str) -> dict:
    """Create training samples from TTS dialogue annotations."""
    with open(annotations_path) as f:
        data = json.load(f)

    stats = {"complete": 0, "incomplete": 0}

    for conv_data in data:
        audio, sr = sf.read(conv_data["audio_path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        turns = conv_data["turns"]
        conv_id = conv_data["conv_id"]

        for i in range(len(turns)):
            turn = turns[i]
            turn_start = turn["start"]
            turn_end = turn["end"]
            turn_dur = turn_end - turn_start

            # Complete at boundaries
            if i > 0:
                boundary_t = turn_start
                end_sample = int(boundary_t * sr)
                start_sample = max(0, end_sample - WINDOW_SAMPLES)
                window = audio[start_sample:end_sample]
                if len(window) >= sr:
                    _save_sample(window, sr, "complete", "nofiller", conv_id, i)
                    stats["complete"] += 1

            # Incomplete mid-turn
            if turn_dur >= 1.5:
                n_points = max(1, int(turn_dur / 1.0))
                for p in range(n_points):
                    frac = (p + 0.5) / (n_points + 1)
                    if frac > 0.8:
                        continue
                    mid_t = turn_start + turn_dur * frac
                    mid_sample = int(mid_t * sr)
                    start_sample = max(0, mid_sample - WINDOW_SAMPLES)
                    window = audio[start_sample:mid_sample]
                    if len(window) >= sr:
                        _save_sample(window, sr, "incomplete", "nofiller", conv_id, i, p)
                        stats["incomplete"] += 1

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    nurc_ann = "data/annotations/nurc_sp_annotations.json"
    tts_ann = "data/annotations/portuguese_tts_annotations.json"

    log.info("Preparing NURC-SP samples...")
    s1 = prepare_from_nurc(nurc_ann)
    log.info("NURC-SP: %s", s1)

    log.info("Preparing TTS samples...")
    s2 = prepare_from_tts(tts_ann)
    log.info("TTS: %s", s2)

    total_complete = s1["complete"] + s2["complete"]
    total_incomplete = s1["incomplete"] + s2["incomplete"]
    log.info("Total: %d complete + %d incomplete = %d samples",
             total_complete, total_incomplete, total_complete + total_incomplete)

    # List output
    import os
    for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
        if filenames:
            log.info("  %s: %d files", os.path.basename(dirpath), len(filenames))
