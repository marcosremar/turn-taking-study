"""
Download and prepare turn-taking evaluation datasets.

Datasets used:
1. Switchboard (HuggingFace) - Two-speaker telephone conversations with timestamps
2. HCRC Map Task (Edinburgh) - Task-oriented dialogues with turn annotations

References:
- Godfrey, J.J., Holliman, E.C., & McDaniel, J. (1992). SWITCHBOARD: Telephone speech
  corpus for research and development. ICASSP-92.
- Anderson, A.H., et al. (1991). The HCRC Map Task Corpus. Language and Speech, 34(4).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SWITCHBOARD_DIR = DATA_DIR / "switchboard"
MAPTASK_DIR = DATA_DIR / "maptask"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


@dataclass
class TurnSegment:
    """A single speaker turn with timing information."""
    speaker: str
    start: float  # seconds
    end: float    # seconds
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Conversation:
    """A conversation with turn-taking annotations."""
    conv_id: str
    audio_path: str
    sample_rate: int
    duration: float  # total duration in seconds
    turns: list[TurnSegment] = field(default_factory=list)
    # Derived labels
    turn_shifts: list[float] = field(default_factory=list)  # timestamps of speaker changes
    holds: list[float] = field(default_factory=list)  # timestamps where same speaker continues after pause


def download_switchboard_from_hf() -> list[Conversation]:
    """Download Switchboard subset from HuggingFace datasets."""
    from datasets import load_dataset

    log.info("Downloading Switchboard from HuggingFace...")
    SWITCHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Use the Switchboard subset available on HF
    try:
        ds = load_dataset("hhoangphuoc/switchboard", split="train", streaming=True)
    except Exception:
        log.warning("HF Switchboard not available, trying alternative...")
        ds = load_dataset("swda", split="train", streaming=True)

    conversations: list[Conversation] = []
    count = 0
    max_conversations = 200  # Limit for benchmark feasibility

    current_conv_id = None
    current_turns: list[TurnSegment] = []

    for sample in ds:
        conv_id = str(sample.get("conversation_no", sample.get("conv_id", count)))

        if conv_id != current_conv_id:
            if current_conv_id is not None and current_turns:
                conv = _build_conversation_from_text(current_conv_id, current_turns)
                if conv:
                    conversations.append(conv)
                    count += 1
                    if count >= max_conversations:
                        break

            current_conv_id = conv_id
            current_turns = []

        speaker = sample.get("caller", sample.get("speaker", "A"))
        text = sample.get("text", sample.get("utterance", ""))
        if text:
            current_turns.append(TurnSegment(
                speaker=str(speaker),
                start=0.0,  # Will be estimated
                end=0.0,
                text=text.strip(),
            ))

    # Save annotations
    _save_annotations(conversations, "switchboard")
    log.info("Downloaded %d Switchboard conversations", len(conversations))
    return conversations


def download_candor_sample() -> list[Conversation]:
    """
    Download CANDOR corpus sample for turn-taking evaluation.

    Reference:
    - Reece, A.G., et al. (2023). The CANDOR corpus: Insights from a large
      multi-modal dataset of naturalistic conversation. Science Advances, 9(13).
    """
    log.info("CANDOR corpus requires manual download from https://cadl.humlab.lu.se/candor/")
    log.info("See: https://www.science.org/doi/10.1126/sciadv.adf3197")
    return []


def generate_synthetic_dataset(
    n_conversations: int = 100,
    min_turns: int = 10,
    max_turns: int = 40,
    sample_rate: int = 16000,
) -> list[Conversation]:
    """
    Generate synthetic two-speaker conversations with ground-truth turn annotations.

    This provides a controlled baseline where we know exact turn boundaries.
    Uses silence/noise segments between speakers to simulate realistic gaps/overlaps.
    """
    log.info("Generating %d synthetic conversations...", n_conversations)
    synth_dir = DATA_DIR / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    conversations = []
    rng = np.random.default_rng(42)

    for i in range(n_conversations):
        n_turns = rng.integers(min_turns, max_turns + 1)
        turns = []
        t = 0.0
        speakers = ["A", "B"]

        for j in range(n_turns):
            speaker = speakers[j % 2]
            # Turn duration: 0.5 - 5.0 seconds
            duration = rng.uniform(0.5, 5.0)
            # Gap between turns: -0.3 (overlap) to 1.5 seconds
            gap = rng.uniform(-0.3, 1.5) if j > 0 else 0.0

            start = max(t + gap, t)  # No negative starts
            end = start + duration

            turns.append(TurnSegment(
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                text=f"[synthetic turn {j}]",
            ))
            t = end

        total_duration = turns[-1].end
        # Generate audio: sine waves at different frequencies per speaker
        n_samples = int(total_duration * sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)

        for turn in turns:
            freq = 200.0 if turn.speaker == "A" else 350.0
            s = int(turn.start * sample_rate)
            e = min(int(turn.end * sample_rate), n_samples)
            t_arr = np.arange(e - s) / sample_rate
            audio[s:e] = 0.3 * np.sin(2 * np.pi * freq * t_arr).astype(np.float32)

        # Add noise
        audio += rng.normal(0, 0.01, n_samples).astype(np.float32)

        audio_path = synth_dir / f"synth_{i:04d}.wav"
        sf.write(str(audio_path), audio, sample_rate)

        # Compute turn shifts and holds
        turn_shifts = []
        holds = []
        for k in range(1, len(turns)):
            if turns[k].speaker != turns[k - 1].speaker:
                turn_shifts.append(turns[k].start)
            else:
                holds.append(turns[k].start)

        conversations.append(Conversation(
            conv_id=f"synth_{i:04d}",
            audio_path=str(audio_path),
            sample_rate=sample_rate,
            duration=total_duration,
            turns=turns,
            turn_shifts=turn_shifts,
            holds=holds,
        ))

    _save_annotations(conversations, "synthetic")
    log.info("Generated %d synthetic conversations (%.1f hours)",
             len(conversations), sum(c.duration for c in conversations) / 3600)
    return conversations


def _build_conversation_from_text(conv_id: str, turns: list[TurnSegment]) -> Conversation | None:
    """Build a Conversation from text-only turns by estimating timing."""
    if len(turns) < 3:
        return None

    # Estimate timing: ~150ms per word + 200ms gap
    t = 0.0
    for i, turn in enumerate(turns):
        words = len(turn.text.split())
        duration = max(0.5, words * 0.15)
        gap = 0.2 if i > 0 else 0.0
        turn.start = round(t + gap, 3)
        turn.end = round(turn.start + duration, 3)
        t = turn.end

    turn_shifts = []
    holds = []
    for k in range(1, len(turns)):
        if turns[k].speaker != turns[k - 1].speaker:
            turn_shifts.append(turns[k].start)
        else:
            holds.append(turns[k].start)

    return Conversation(
        conv_id=conv_id,
        audio_path="",  # text-only
        sample_rate=16000,
        duration=turns[-1].end,
        turns=turns,
        turn_shifts=turn_shifts,
        holds=holds,
    )


def _save_annotations(conversations: list[Conversation], name: str) -> None:
    """Save conversation annotations to JSON for reproducibility."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for conv in conversations:
        out.append({
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

    path = ANNOTATIONS_DIR / f"{name}_annotations.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved %d annotations to %s", len(out), path)


def load_annotations(name: str) -> list[Conversation]:
    """Load previously saved annotations."""
    path = ANNOTATIONS_DIR / f"{name}_annotations.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotations not found: {path}")

    with open(path) as f:
        data = json.load(f)

    conversations = []
    for item in data:
        turns = [TurnSegment(**t) for t in item["turns"]]
        conversations.append(Conversation(
            conv_id=item["conv_id"],
            audio_path=item["audio_path"],
            sample_rate=item["sample_rate"],
            duration=item["duration"],
            turns=turns,
            turn_shifts=item["turn_shifts"],
            holds=item["holds"],
        ))
    return conversations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download turn-taking datasets")
    parser.add_argument("--dataset", choices=["switchboard", "synthetic", "all"], default="all")
    parser.add_argument("--n-synthetic", type=int, default=100)
    args = parser.parse_args()

    if args.dataset in ("synthetic", "all"):
        generate_synthetic_dataset(n_conversations=args.n_synthetic)

    if args.dataset in ("switchboard", "all"):
        download_switchboard_from_hf()
