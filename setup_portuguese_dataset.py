"""
Download and prepare Portuguese conversation datasets for turn-taking evaluation.

Datasets:
1. NURC-SP / CORAL-BRASIL — Brazilian Portuguese spontaneous dialogue
2. Common Voice PT — Mozilla, single speaker (for baseline audio)
3. Synthetic Portuguese — generated with controlled turn timing

References:
- Castilho, A.T. (2019). NURC-SP Audio Corpus. 239h of transcribed
  Brazilian Portuguese dialogues.
- ASR-BPCSC: Brazilian Portuguese Conversational Speech Corpus.
  10h transcribed conversational speech, 30 conversations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PT_DIR = DATA_DIR / "portuguese"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


@dataclass
class TurnSegment:
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Conversation:
    conv_id: str
    audio_path: str
    sample_rate: int
    duration: float
    turns: list[TurnSegment] = field(default_factory=list)
    turn_shifts: list[float] = field(default_factory=list)
    holds: list[float] = field(default_factory=list)


def download_common_voice_pt_dialogues(max_pairs: int = 50) -> list[Conversation]:
    """
    Download Common Voice Portuguese and create synthetic dialogues
    by concatenating different speakers' utterances.
    """
    from datasets import load_dataset

    log.info("Downloading Common Voice Portuguese samples...")
    PT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "pt",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning("Common Voice requires login. Trying alternative: %s", e)
        return []

    # Collect samples from different speakers
    speaker_samples: dict[str, list] = {}
    count = 0
    for sample in ds:
        client_id = sample.get("client_id", str(count))
        if client_id not in speaker_samples:
            speaker_samples[client_id] = []
        if len(speaker_samples[client_id]) < 10:
            speaker_samples[client_id].append({
                "audio": sample["audio"]["array"],
                "sr": sample["audio"]["sampling_rate"],
                "text": sample.get("sentence", ""),
            })
        count += 1
        if len(speaker_samples) >= 20 and all(len(v) >= 3 for v in speaker_samples.values()):
            break
        if count > 5000:
            break

    # Create dialogue pairs
    conversations = _create_dialogues_from_speakers(speaker_samples, max_pairs)
    _save_annotations(conversations, "portuguese_cv")
    return conversations


def generate_portuguese_synthetic(
    n_conversations: int = 100,
    min_turns: int = 8,
    max_turns: int = 30,
    sample_rate: int = 16000,
) -> list[Conversation]:
    """
    Generate synthetic Portuguese dialogues with precise turn annotations.

    Simulates realistic Portuguese conversation patterns:
    - Average turn duration: 1.5-4s (Portuguese speakers tend to have longer turns)
    - Inter-turn gap: median ~200ms (typical for Portuguese)
    - Overlap rate: ~15% of turns (Portuguese has more overlap than English)
    """
    log.info("Generating %d synthetic Portuguese conversations...", n_conversations)
    synth_dir = PT_DIR / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    conversations = []
    rng = np.random.default_rng(42)

    # Portuguese conversation timing parameters
    # Based on NURC-SP and C-ORAL-BRASIL studies
    turn_duration_mean = 2.5  # seconds
    turn_duration_std = 1.2
    gap_mean = 0.2  # seconds (Portuguese has shorter gaps)
    gap_std = 0.4
    overlap_prob = 0.15  # 15% overlap rate

    portuguese_phrases = [
        "Olha, eu acho que isso faz sentido",
        "Pois é, mas tem outro ponto importante",
        "Concordo plenamente com você",
        "Não sei se entendi bem",
        "Vamos ver como funciona na prática",
        "Isso é interessante, mas...",
        "Exatamente, é isso mesmo",
        "Deixa eu pensar um pouco",
        "Bom, na minha opinião",
        "Então, o que você acha?",
        "Sim, sim, com certeza",
        "Espera, deixa eu falar",
        "Tá bom, entendi",
        "Mas olha só uma coisa",
        "É verdade, faz sentido",
        "Ah, interessante",
        "Hmm, não tenho certeza",
        "Pode ser, pode ser",
        "Legal, vamos continuar",
        "Enfim, voltando ao assunto",
    ]

    for i in range(n_conversations):
        n_turns = rng.integers(min_turns, max_turns + 1)
        turns = []
        t = 0.0
        speakers = ["A", "B"]

        hold_prob = 0.25  # 25% chance same speaker continues (hold)
        prev_speaker = None
        for j in range(n_turns):
            if prev_speaker is None or rng.random() >= hold_prob:
                speaker = speakers[j % 2]  # Normal alternation
            else:
                speaker = prev_speaker  # Same speaker continues (hold)

            # Turn duration with Portuguese distribution
            duration = max(0.4, rng.normal(turn_duration_mean, turn_duration_std))

            # Gap (can be negative for overlap)
            if j > 0:
                if rng.random() < overlap_prob:
                    gap = rng.uniform(-0.5, -0.05)  # Overlap
                else:
                    gap = max(0.05, rng.normal(gap_mean, gap_std))
            else:
                gap = 0.0

            start = max(t + gap, 0.0)
            end = start + duration
            text = rng.choice(portuguese_phrases)

            turns.append(TurnSegment(
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
            ))
            prev_speaker = speaker
            t = end

        total_duration = turns[-1].end

        # Generate stereo speech-like audio (ch0=speaker A, ch1=speaker B)
        # Uses filtered noise + harmonics to simulate speech formants
        n_samples = int(total_duration * sample_rate)
        audio_a = np.zeros(n_samples, dtype=np.float32)
        audio_b = np.zeros(n_samples, dtype=np.float32)

        for turn in turns:
            f0 = 130.0 if turn.speaker == "A" else 200.0
            s = int(turn.start * sample_rate)
            e = min(int(turn.end * sample_rate), n_samples)
            dur = e - s
            if dur <= 0:
                continue

            t_arr = np.arange(dur) / sample_rate

            # Glottal pulse train (harmonics simulate voiced speech)
            harmonics = np.zeros(dur, dtype=np.float32)
            for h in range(1, 8):
                amp = 0.3 / h  # Falling spectral envelope
                jitter = rng.uniform(0.98, 1.02)  # Pitch jitter
                harmonics += amp * np.sin(2 * np.pi * f0 * h * jitter * t_arr).astype(np.float32)

            # Aspiration noise (unvoiced component)
            noise = rng.normal(0, 0.08, dur).astype(np.float32)

            # Formant-like bandpass: weight low freqs more (speech is 300-3000Hz)
            from scipy.signal import butter, lfilter
            b_low, a_low = butter(2, [200, 3500], btype='band', fs=sample_rate)
            noise_filtered = lfilter(b_low, a_low, noise).astype(np.float32)

            signal = harmonics * 0.7 + noise_filtered * 0.3

            # Amplitude modulation (syllable rhythm ~4-5Hz for Portuguese)
            syllable_rate = rng.uniform(4.0, 5.5)
            modulation = 0.6 + 0.4 * np.sin(2 * np.pi * syllable_rate * t_arr).astype(np.float32)
            signal *= modulation

            # Envelope with natural attack/release
            envelope = np.ones(dur, dtype=np.float32)
            attack = min(int(0.03 * sample_rate), dur // 4)
            release = min(int(0.06 * sample_rate), dur // 4)
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack).astype(np.float32)
            if release > 0:
                envelope[-release:] = np.linspace(1, 0, release).astype(np.float32)

            target = audio_a if turn.speaker == "A" else audio_b
            target[s:e] += signal * envelope

        # Low ambient noise on both channels
        audio_a += rng.normal(0, 0.003, n_samples).astype(np.float32)
        audio_b += rng.normal(0, 0.003, n_samples).astype(np.float32)
        audio_a = np.clip(audio_a, -1.0, 1.0)
        audio_b = np.clip(audio_b, -1.0, 1.0)
        # Also save mono mix for models that expect mono
        audio = (audio_a + audio_b) / 2.0

        # Save stereo (for VAP) and mono (for VAD/silence)
        stereo = np.stack([audio_a, audio_b], axis=-1)  # (samples, 2)
        audio_path_stereo = synth_dir / f"pt_synth_{i:04d}_stereo.wav"
        sf.write(str(audio_path_stereo), stereo, sample_rate)

        audio_path = synth_dir / f"pt_synth_{i:04d}.wav"
        sf.write(str(audio_path), audio, sample_rate)

        # Compute turn events
        turn_shifts = []
        holds = []
        for k in range(1, len(turns)):
            if turns[k].speaker != turns[k - 1].speaker:
                turn_shifts.append(turns[k].start)
            else:
                holds.append(turns[k].start)

        conversations.append(Conversation(
            conv_id=f"pt_synth_{i:04d}",
            audio_path=str(audio_path),
            sample_rate=sample_rate,
            duration=total_duration,
            turns=turns,
            turn_shifts=turn_shifts,
            holds=holds,
        ))

    _save_annotations(conversations, "portuguese_synthetic")
    total_hours = sum(c.duration for c in conversations) / 3600
    log.info("Generated %d Portuguese conversations (%.1f hours)", len(conversations), total_hours)
    return conversations


def _create_dialogues_from_speakers(
    speaker_samples: dict[str, list],
    max_pairs: int,
) -> list[Conversation]:
    """Create dialogues by interleaving samples from different speakers."""
    conversations = []
    speakers = list(speaker_samples.keys())
    rng = np.random.default_rng(123)

    for pair_idx in range(min(max_pairs, len(speakers) // 2)):
        sp_a = speakers[pair_idx * 2]
        sp_b = speakers[pair_idx * 2 + 1]
        samples_a = speaker_samples[sp_a]
        samples_b = speaker_samples[sp_b]

        turns = []
        audio_chunks = []
        t = 0.0
        target_sr = 16000

        n_turns = min(len(samples_a) + len(samples_b), 10)
        for j in range(n_turns):
            if j % 2 == 0 and samples_a:
                sample = samples_a.pop(0)
                speaker = "A"
            elif samples_b:
                sample = samples_b.pop(0)
                speaker = "B"
            else:
                break

            audio = np.array(sample["audio"], dtype=np.float32)
            sr = sample["sr"]

            # Resample if needed
            if sr != target_sr:
                import torchaudio
                import torch
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                tensor = resampler(tensor)
                audio = tensor.squeeze().numpy()

            duration = len(audio) / target_sr
            gap = rng.uniform(0.1, 0.5) if j > 0 else 0.0

            # Add gap silence
            if gap > 0:
                audio_chunks.append(np.zeros(int(gap * target_sr), dtype=np.float32))

            start = t + gap
            end = start + duration

            turns.append(TurnSegment(
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                text=sample.get("text", ""),
            ))
            audio_chunks.append(audio)
            t = end

        if len(turns) < 3:
            continue

        # Concatenate audio
        full_audio = np.concatenate(audio_chunks)
        audio_path = PT_DIR / f"cv_dialogue_{pair_idx:04d}.wav"
        sf.write(str(audio_path), full_audio, target_sr)

        turn_shifts = []
        holds = []
        for k in range(1, len(turns)):
            if turns[k].speaker != turns[k - 1].speaker:
                turn_shifts.append(turns[k].start)
            else:
                holds.append(turns[k].start)

        conversations.append(Conversation(
            conv_id=f"cv_dialogue_{pair_idx:04d}",
            audio_path=str(audio_path),
            sample_rate=target_sr,
            duration=turns[-1].end,
            turns=turns,
            turn_shifts=turn_shifts,
            holds=holds,
        ))

    return conversations


def _save_annotations(conversations: list[Conversation], name: str) -> None:
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
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("Saved %d annotations to %s", len(out), path)


def load_annotations(name: str) -> list[Conversation]:
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "common_voice", "all"], default="all")
    parser.add_argument("--n-synthetic", type=int, default=100)
    args = parser.parse_args()

    if args.dataset in ("synthetic", "all"):
        generate_portuguese_synthetic(n_conversations=args.n_synthetic)

    if args.dataset in ("common_voice", "all"):
        download_common_voice_pt_dialogues()
