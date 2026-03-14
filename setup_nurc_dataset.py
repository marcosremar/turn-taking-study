"""
Prepare NURC-SP Corpus Minimo dialogues for turn-taking benchmark.

Reconstructs continuous audio from segmented audio files and builds
Conversation objects with ground truth turn annotations.

Dataset: NURC-SP Corpus Minimo (nilc-nlp/NURC-SP_Corpus_Minimo on HuggingFace)
- Real Brazilian Portuguese spontaneous dialogues from the 1970s-1990s
- Manually annotated speaker turns with timestamps
- CC BY-NC-ND 4.0 license
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from setup_portuguese_dataset import Conversation, TurnSegment

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SEGMENTS_DIR = DATA_DIR / "nurc_sp" / "segmented_audios"
NURC_DIR = DATA_DIR / "nurc_sp"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Only use multi-speaker dialogues (DID = diálogos, D2 = diálogos entre informantes)
DIALOGUE_PREFIXES = ("SP_DID_", "SP_D2_")

TARGET_SR = 16000


def load_nurc_metadata() -> pd.DataFrame:
    """Load NURC-SP segment metadata."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "nilc-nlp/NURC-SP_Corpus_Minimo",
        "segmented_audios_time.csv",
        repo_type="dataset",
    )
    return pd.read_csv(path)


def build_conversation(name: str, segments_df: pd.DataFrame) -> Conversation | None:
    """Reconstruct continuous audio and build Conversation from segments."""
    # Sort segments by start_time
    segments_df = segments_df.sort_values("start_time").reset_index(drop=True)

    # Filter to segments that have audio files
    seg_dir = SEGMENTS_DIR / name
    if not seg_dir.exists():
        log.warning("No audio directory for %s", name)
        return None

    # Build timeline: read each segment's audio and place at correct offset
    total_end = segments_df["end_time"].max()
    total_start = segments_df["start_time"].min()
    duration = total_end - total_start

    # Limit to first 5 minutes for benchmark speed
    max_duration = 300.0  # 5 minutes
    if duration > max_duration:
        cutoff = total_start + max_duration
        segments_df = segments_df[segments_df["start_time"] < cutoff].copy()
        total_end = min(cutoff, segments_df["end_time"].max())
        duration = total_end - total_start

    n_samples = int(duration * TARGET_SR) + TARGET_SR  # Extra second buffer
    audio = np.zeros(n_samples, dtype=np.float32)

    turns = []
    loaded = 0
    skipped = 0

    for _, row in segments_df.iterrows():
        # Find the audio file
        start_str = f"{row['start_time']:.2f}"
        end_str = f"{row['end_time']:.2f}"
        pattern = f"{name}_seg_{start_str}_{end_str}.wav"
        audio_path = seg_dir / pattern

        if not audio_path.exists():
            # Try matching with different decimal precision
            candidates = list(seg_dir.glob(f"{name}_seg_{start_str[:5]}*_{end_str[:5]}*.wav"))
            if candidates:
                audio_path = candidates[0]
            else:
                skipped += 1
                continue

        try:
            seg_audio, sr = sf.read(str(audio_path))
        except Exception:
            skipped += 1
            continue

        if seg_audio.ndim > 1:
            seg_audio = seg_audio.mean(axis=1)
        seg_audio = seg_audio.astype(np.float32)

        # Resample if needed
        if sr != TARGET_SR:
            import torchaudio
            import torch
            tensor = torch.from_numpy(seg_audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            tensor = resampler(tensor)
            seg_audio = tensor.squeeze().numpy()

        # Place in timeline
        offset = row["start_time"] - total_start
        start_idx = int(offset * TARGET_SR)
        end_idx = start_idx + len(seg_audio)

        if end_idx > len(audio):
            seg_audio = seg_audio[:len(audio) - start_idx]
            end_idx = len(audio)

        if start_idx < len(audio) and len(seg_audio) > 0:
            audio[start_idx:start_idx + len(seg_audio)] = seg_audio
            loaded += 1

        # Create turn segment
        turns.append(TurnSegment(
            speaker=row["speaker"],
            start=round(offset, 3),
            end=round(offset + (row["end_time"] - row["start_time"]), 3),
            text=str(row.get("normalized_text", "")),
        ))

    if loaded < 5:
        log.warning("Only loaded %d/%d segments for %s, skipping", loaded, loaded + skipped, name)
        return None

    # Trim audio to actual content
    actual_end = max(t.end for t in turns) if turns else 0
    n_samples = int(actual_end * TARGET_SR) + TARGET_SR
    audio = audio[:n_samples]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    # Save reconstructed audio
    NURC_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = NURC_DIR / f"{name}.wav"
    sf.write(str(audio_path), audio, TARGET_SR)

    # Compute events
    turn_shifts = []
    holds = []
    for k in range(1, len(turns)):
        if turns[k].speaker != turns[k - 1].speaker:
            turn_shifts.append(turns[k].start)
        else:
            holds.append(turns[k].start)

    log.info("  %s: %d turns (%d loaded, %d skipped), %d shifts, %d holds, %.0fs",
             name, len(turns), loaded, skipped, len(turn_shifts), len(holds), actual_end)

    return Conversation(
        conv_id=name,
        audio_path=str(audio_path),
        sample_rate=TARGET_SR,
        duration=actual_end,
        turns=turns,
        turn_shifts=turn_shifts,
        holds=holds,
    )


def prepare_nurc_dataset(conversation_names: list[str] | None = None) -> list[Conversation]:
    """Prepare NURC-SP conversations for benchmarking."""
    df = load_nurc_metadata()

    # Filter to dialogues only
    dialogue_df = df[df["name"].str.startswith(DIALOGUE_PREFIXES)]
    available_names = sorted(dialogue_df["name"].unique())

    if conversation_names:
        names = [n for n in conversation_names if n in available_names]
    else:
        names = available_names

    log.info("Preparing %d NURC-SP conversations...", len(names))

    conversations = []
    for name in names:
        conv_df = dialogue_df[dialogue_df["name"] == name]
        conv = build_conversation(name, conv_df)
        if conv:
            conversations.append(conv)

    # Save annotations
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ann_data = []
    for conv in conversations:
        ann_data.append({
            "conv_id": conv.conv_id,
            "audio_path": conv.audio_path,
            "sample_rate": conv.sample_rate,
            "duration": conv.duration,
            "n_turns": len(conv.turns),
            "n_turn_shifts": len(conv.turn_shifts),
            "n_holds": len(conv.holds),
            "turns": [
                {"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text}
                for t in conv.turns
            ],
            "turn_shifts": conv.turn_shifts,
            "holds": conv.holds,
        })

    ann_path = ANNOTATIONS_DIR / "nurc_sp_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(ann_data, f, indent=2, ensure_ascii=False)

    total_hours = sum(c.duration for c in conversations) / 3600
    total_shifts = sum(len(c.turn_shifts) for c in conversations)
    total_holds = sum(len(c.holds) for c in conversations)
    log.info("Prepared %d conversations: %.1f min, %d shifts, %d holds",
             len(conversations), total_hours * 60, total_shifts, total_holds)

    return conversations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_nurc_dataset()
