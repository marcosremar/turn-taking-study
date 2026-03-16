"""Generate TTS audio for the turn-taking dataset.

Based on the Pipecat v3.1 pipeline:
1. Native PT-BR voices via Kokoro (pf_dora, pm_alex, pm_santa)
2. French-accented Portuguese via XTTS v2 voice cloning
3. Speed/pitch variation + noise augmentation
4. Hesitation pause injection (1.5-3s silence after fillers) — SpeculativeETD V3
5. Short utterance dataset ("sim", "nao", "ok") — Pipecat v3.2 (-40% errors)

Pipecat used Google Chirp3 TTS; we use Kokoro (open-source, runs locally)
+ XTTS v2 (voice cloning for accent simulation).

Run locally or on Modal GPU:
    python 03_generate_audio.py
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_RATE = 16000  # Pipecat model expects 16kHz


@dataclass
class AudioSample:
    """A single audio sample for the dataset."""
    audio: np.ndarray      # float32, 16kHz
    text: str
    label: str             # "complete" or "incomplete"
    voice: str             # e.g. "pf_dora", "fr_clone_1"
    accent: str            # "native_pt_br" or "french_pt"
    source: str            # "kokoro", "xtts", "pipecat"
    speed: float = 1.0


# ---------------------------------------------------------------------------
# Kokoro TTS — native PT-BR voices
# ---------------------------------------------------------------------------

def generate_kokoro_audio(
    sentences: list[dict],
    voices: list[str] | None = None,
    speed_range: tuple[float, float] = (0.85, 1.15),
) -> list[AudioSample]:
    """Generate audio using Kokoro TTS (PT-BR voices).

    sentences: list of {"text": str, "label": "complete"|"incomplete"}
    voices: Kokoro voice IDs (default: PT-BR voices)
    """
    try:
        from kokoro import KPipeline
    except ImportError:
        log.error("kokoro not installed — pip install kokoro")
        return []

    if voices is None:
        voices = ["pf_dora", "pm_alex", "pm_santa"]

    log.info("Initializing Kokoro pipeline (lang=pt)...")
    pipeline = KPipeline(lang_code="p")  # Portuguese

    samples = []
    errors = 0

    for i, sent in enumerate(sentences):
        text = sent["text"]
        label = sent["label"]
        voice = random.choice(voices)
        speed = random.uniform(*speed_range)

        try:
            # Generate audio
            generator = pipeline(text, voice=voice, speed=speed)
            audio_chunks = []
            for _, _, chunk in generator:
                audio_chunks.append(chunk.numpy() if hasattr(chunk, 'numpy') else np.array(chunk))

            if not audio_chunks:
                errors += 1
                continue

            audio = np.concatenate(audio_chunks).astype(np.float32)

            # Resample to 16kHz if needed (Kokoro outputs 24kHz)
            if hasattr(pipeline, 'sample_rate') and pipeline.sample_rate != SAMPLE_RATE:
                import torchaudio
                import torch
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(
                    tensor, pipeline.sample_rate, SAMPLE_RATE
                ).squeeze().numpy()

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9

            samples.append(AudioSample(
                audio=audio,
                text=text,
                label=label,
                voice=voice,
                accent="native_pt_br",
                source="kokoro",
                speed=speed,
            ))

        except Exception as e:
            errors += 1
            if errors <= 5:
                log.warning("Kokoro error on sentence %d: %s", i, e)

        if (i + 1) % 100 == 0:
            log.info("  Kokoro: %d/%d generated (%d errors)", len(samples), i + 1, errors)

    log.info("Kokoro: %d samples generated (%d errors)", len(samples), errors)
    return samples


# ---------------------------------------------------------------------------
# XTTS v2 — French-accented Portuguese via voice cloning
# ---------------------------------------------------------------------------

def generate_xtts_audio(
    sentences: list[dict],
    reference_audio_dir: Path | None = None,
    speed_range: tuple[float, float] = (0.9, 1.1),
) -> list[AudioSample]:
    """Generate audio with French accent using XTTS v2 voice cloning.

    Uses short French speech samples as reference to clone the accent,
    then synthesizes Portuguese text with that voice.

    sentences: list of {"text": str, "label": "complete"|"incomplete"}
    reference_audio_dir: directory with .wav files of French speakers
    """
    try:
        from TTS.api import TTS
    except ImportError:
        log.error("TTS (coqui) not installed — pip install TTS")
        return []

    log.info("Initializing XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # Find reference audio files (French speakers)
    ref_files = []
    if reference_audio_dir and reference_audio_dir.exists():
        ref_files = list(reference_audio_dir.glob("*.wav"))

    if not ref_files:
        log.warning("No French reference audio found in %s — using default voice", reference_audio_dir)
        # Fall back to generating without cloning (still works, but no accent)
        return _generate_xtts_no_clone(tts, sentences, speed_range)

    log.info("Found %d French reference voices", len(ref_files))

    samples = []
    errors = 0

    for i, sent in enumerate(sentences):
        text = sent["text"]
        label = sent["label"]
        ref = random.choice(ref_files)

        try:
            audio = tts.tts(
                text=text,
                speaker_wav=str(ref),
                language="pt",
            )
            audio = np.array(audio, dtype=np.float32)

            # Resample to 16kHz if needed
            if hasattr(tts, 'synthesizer') and tts.synthesizer.output_sample_rate != SAMPLE_RATE:
                import torchaudio
                import torch
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(
                    tensor, tts.synthesizer.output_sample_rate, SAMPLE_RATE
                ).squeeze().numpy()

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9

            # Random speed variation
            speed = random.uniform(*speed_range)
            if abs(speed - 1.0) > 0.05:
                indices = np.arange(0, len(audio), speed).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            samples.append(AudioSample(
                audio=audio,
                text=text,
                label=label,
                voice=f"fr_clone_{ref.stem}",
                accent="french_pt",
                source="xtts",
                speed=speed,
            ))

        except Exception as e:
            errors += 1
            if errors <= 5:
                log.warning("XTTS error on sentence %d: %s", i, e)

        if (i + 1) % 50 == 0:
            log.info("  XTTS: %d/%d generated (%d errors)", len(samples), i + 1, errors)

    log.info("XTTS: %d samples generated (%d errors)", len(samples), errors)
    return samples


def _generate_xtts_no_clone(
    tts,
    sentences: list[dict],
    speed_range: tuple[float, float],
) -> list[AudioSample]:
    """XTTS generation without voice cloning (fallback)."""
    samples = []
    errors = 0

    for i, sent in enumerate(sentences):
        try:
            audio = tts.tts(text=sent["text"], language="pt")
            audio = np.array(audio, dtype=np.float32)

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9

            samples.append(AudioSample(
                audio=audio,
                text=sent["text"],
                label=sent["label"],
                voice="xtts_default",
                accent="french_pt",
                source="xtts",
                speed=1.0,
            ))
        except Exception as e:
            errors += 1
            if errors <= 3:
                log.warning("XTTS fallback error: %s", e)

    log.info("XTTS (no clone): %d samples (%d errors)", len(samples), errors)
    return samples


# ---------------------------------------------------------------------------
# Hesitation pause injection (SpeculativeETD V3 — best data variant)
# ---------------------------------------------------------------------------

def inject_hesitation_pause(
    audio: np.ndarray,
    pause_duration_range: tuple[float, float] = (1.5, 3.0),
    position: str = "end",
) -> np.ndarray:
    """Inject a realistic hesitation pause into audio.

    SpeculativeETD V3 showed that inserting 1.5-3.0s pauses after fillers
    was the most effective data variant for training ETD models.

    For French speakers learning Portuguese, these long pauses happen when:
    - Searching for a word ("Eu preciso de... [2s pause] ...uma tesoura")
    - Thinking about conjugation ("Ontem eu... [2s pause] ...fui ao mercado")
    - Code-switching hesitation ("Eu gosto de... [1.5s] ...euh... praia")

    position: "end" = pause at end (simulates mid-utterance stop)
              "mid" = pause in the middle (simulates hesitation)
    """
    pause_s = random.uniform(*pause_duration_range)
    pause_samples = int(pause_s * SAMPLE_RATE)

    # Add very low-level noise to the pause (not pure silence — more realistic)
    pause = np.random.randn(pause_samples).astype(np.float32) * 0.001

    if position == "end":
        # Pause at end: speaker stops mid-sentence (INCOMPLETE)
        return np.concatenate([audio, pause])
    else:
        # Pause in middle: split audio and insert pause
        if len(audio) < SAMPLE_RATE:  # too short to split
            return np.concatenate([audio, pause])
        split_point = random.randint(len(audio) // 3, 2 * len(audio) // 3)
        return np.concatenate([audio[:split_point], pause, audio[split_point:]])


def create_hesitation_variants(
    samples: list[AudioSample],
    fraction: float = 0.3,
) -> list[AudioSample]:
    """Create hesitation-pause variants from existing INCOMPLETE samples.

    Takes a fraction of incomplete samples and adds 1.5-3s pauses,
    simulating French speakers hesitating in Portuguese.
    """
    incomplete = [s for s in samples if s.label == "incomplete"]
    n_variants = int(len(incomplete) * fraction)

    variants = []
    for s in random.sample(incomplete, min(n_variants, len(incomplete))):
        # Variant 1: long pause at end (speaker stopped to think)
        audio_end = inject_hesitation_pause(s.audio, position="end")
        variants.append(AudioSample(
            audio=audio_end,
            text=s.text,
            label="incomplete",  # still incomplete — speaker will continue
            voice=s.voice,
            accent=s.accent,
            source=f"{s.source}_hesitation_end",
            speed=s.speed,
        ))

        # Variant 2: pause in the middle (thinking mid-sentence)
        if random.random() < 0.5:
            audio_mid = inject_hesitation_pause(s.audio, position="mid")
            variants.append(AudioSample(
                audio=audio_mid,
                text=s.text,
                label="incomplete",
                voice=s.voice,
                accent=s.accent,
                source=f"{s.source}_hesitation_mid",
                speed=s.speed,
            ))

    log.info("Created %d hesitation-pause variants from %d incomplete samples",
             len(variants), len(incomplete))
    return variants


# ---------------------------------------------------------------------------
# Short utterance generation (Pipecat v3.2: -40% errors on short responses)
# ---------------------------------------------------------------------------

# Common short Portuguese responses in meetings
SHORT_UTTERANCES_COMPLETE = [
    "Sim.", "Não.", "Ok.", "Tá.", "Pode.", "Beleza.", "Certo.", "Claro.",
    "Entendi.", "Combinado.", "Perfeito.", "Exato.", "Isso.", "Verdade.",
    "Com certeza.", "Sem dúvida.", "Tá bom.", "Pode ser.", "Vamos lá.",
    "Concordo.", "Fechado.", "Ótimo.", "Legal.", "Tranquilo.", "Valeu.",
    "Obrigado.", "De nada.", "Até logo.", "Tchau.", "Bom dia.", "Boa tarde.",
    # French speaker variants
    "Oui... quer dizer, sim.", "Sim, euh, concordo.", "Ok, d'accord.",
    "Bon, tá bom.", "Voilà, é isso.", "Exactement, exato.",
]

SHORT_UTTERANCES_INCOMPLETE = [
    "Sim, mas...", "Não, porque...", "Então...", "Tipo...", "Olha...",
    "Bom...", "Na verdade...", "Quer dizer...", "Pois é...", "Sabe...",
    "É que...", "O problema é...", "A questão é...", "Eu acho que...",
    # French speaker variants (hesitating after short start)
    "Oui... euh...", "Sim, mas... comment dire...", "Alors...",
    "Bon, eu acho que... euh...", "C'est... quer dizer...",
    "Não, enfin... na verdade...", "Sim, mas... como se diz...",
]


def generate_short_utterances(
    voices: list[str] | None = None,
    n_per_utterance: int = 3,
) -> list[AudioSample]:
    """Generate audio for short utterances using Kokoro TTS.

    Pipecat v3.2 showed that a dedicated short utterance dataset
    reduced misclassification by 40%. Short responses like "sim", "não"
    are very common in Portuguese meetings.
    """
    try:
        from kokoro import KPipeline
    except ImportError:
        log.warning("kokoro not installed — skipping short utterances")
        return []

    if voices is None:
        voices = ["pf_dora", "pm_alex", "pm_santa"]

    pipeline = KPipeline(lang_code="p")
    samples = []
    errors = 0

    all_short = (
        [(text, "complete") for text in SHORT_UTTERANCES_COMPLETE]
        + [(text, "incomplete") for text in SHORT_UTTERANCES_INCOMPLETE]
    )

    for text, label in all_short:
        for _ in range(n_per_utterance):
            voice = random.choice(voices)
            speed = random.uniform(0.85, 1.15)

            try:
                generator = pipeline(text, voice=voice, speed=speed)
                audio_chunks = []
                for _, _, chunk in generator:
                    audio_chunks.append(chunk.numpy() if hasattr(chunk, 'numpy') else np.array(chunk))

                if not audio_chunks:
                    errors += 1
                    continue

                audio = np.concatenate(audio_chunks).astype(np.float32)

                # Resample to 16kHz if needed
                if hasattr(pipeline, 'sample_rate') and pipeline.sample_rate != SAMPLE_RATE:
                    import torchaudio
                    import torch
                    tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    audio = torchaudio.functional.resample(
                        tensor, pipeline.sample_rate, SAMPLE_RATE
                    ).squeeze().numpy()

                # Normalize
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.9

                # For incomplete short utterances, add hesitation pause
                if label == "incomplete" and random.random() < 0.7:
                    audio = inject_hesitation_pause(audio, pause_duration_range=(1.0, 2.5), position="end")

                samples.append(AudioSample(
                    audio=audio,
                    text=text,
                    label=label,
                    voice=voice,
                    accent="native_pt_br",
                    source="short_utterance",
                    speed=speed,
                ))

            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning("Short utterance error: %s", e)

    n_c = sum(1 for s in samples if s.label == "complete")
    n_i = sum(1 for s in samples if s.label == "incomplete")
    log.info("Short utterances: %d samples (%d complete, %d incomplete, %d errors)",
             len(samples), n_c, n_i, errors)
    return samples


# ---------------------------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------------------------

def augment_sample(audio: np.ndarray) -> np.ndarray:
    """Apply augmentation to diversify training data.

    Based on Pipecat/SpeculativeETD augmentation strategies.
    """
    aug = audio.copy()

    # Add background noise (simulates meeting room)
    if random.random() < 0.4:
        noise_level = random.uniform(0.002, 0.015)
        aug += np.random.randn(len(aug)).astype(np.float32) * noise_level

    # Volume variation (simulates different mic distances)
    if random.random() < 0.5:
        scale = random.uniform(0.6, 1.4)
        aug *= scale

    # Speed perturbation (simulates speaking rate variation)
    if random.random() < 0.3:
        speed = random.uniform(0.92, 1.08)
        indices = np.arange(0, len(aug), speed).astype(int)
        indices = indices[indices < len(aug)]
        aug = aug[indices]

    return np.clip(aug, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------

def save_dataset(
    samples: list[AudioSample],
    output_dir: Path,
    augment_copies: int = 2,
) -> Path:
    """Save audio samples as WAV files + metadata JSON.

    augment_copies: number of augmented copies per sample (0 = no augmentation)
    """
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    metadata = []
    total_saved = 0

    for i, sample in enumerate(samples):
        # Save original
        fname = f"{i:05d}_{sample.label}_{sample.accent}_{sample.voice}.wav"
        sf.write(str(audio_dir / fname), sample.audio, SAMPLE_RATE)
        metadata.append({
            "file": fname,
            "text": sample.text,
            "label": sample.label,
            "voice": sample.voice,
            "accent": sample.accent,
            "source": sample.source,
            "speed": sample.speed,
            "augmented": False,
            "duration_s": round(len(sample.audio) / SAMPLE_RATE, 2),
        })
        total_saved += 1

        # Save augmented copies
        for aug_idx in range(augment_copies):
            aug_audio = augment_sample(sample.audio)
            aug_fname = f"{i:05d}_{sample.label}_{sample.accent}_{sample.voice}_aug{aug_idx}.wav"
            sf.write(str(audio_dir / aug_fname), aug_audio, SAMPLE_RATE)
            metadata.append({
                "file": aug_fname,
                "text": sample.text,
                "label": sample.label,
                "voice": sample.voice,
                "accent": sample.accent,
                "source": sample.source,
                "speed": sample.speed,
                "augmented": True,
                "duration_s": round(len(aug_audio) / SAMPLE_RATE, 2),
            })
            total_saved += 1

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Stats
    n_complete = sum(1 for m in metadata if m["label"] == "complete")
    n_incomplete = sum(1 for m in metadata if m["label"] == "incomplete")
    n_native = sum(1 for m in metadata if m["accent"] == "native_pt_br")
    n_french = sum(1 for m in metadata if m["accent"] == "french_pt")

    log.info("Dataset saved to %s:", output_dir)
    log.info("  Total: %d samples (%d original + %d augmented)",
             total_saved, len(samples), total_saved - len(samples))
    log.info("  Complete: %d, Incomplete: %d", n_complete, n_incomplete)
    log.info("  Native PT-BR: %d, French-accented: %d", n_native, n_french)

    return output_dir


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_audio_generation(
    max_native_sentences: int = 3000,
    max_french_sentences: int = 1000,
    augment_copies: int = 2,
) -> Path:
    """Run the full audio generation pipeline.

    Loads labeled sentences from 02_generate_labels.py output,
    generates audio via Kokoro + XTTS, saves dataset.
    """
    labeled_dir = DATA_DIR / "claude_labeled"

    # Load labeled sentences
    all_sentences = []

    # 1. Classified sentences (from CORAA transcripts)
    classified_path = labeled_dir / "classified_pt.json"
    if classified_path.exists():
        with open(classified_path) as f:
            classified = json.load(f)
        for c in classified:
            if c.get("label") in ("completo", "incompleto"):
                all_sentences.append({
                    "text": c["text"],
                    "label": "complete" if c["label"] == "completo" else "incomplete",
                    "source": "classified",
                })
        log.info("Loaded %d classified sentences", len(all_sentences))

    # 2. Filler sentences (all are INCOMPLETE)
    for filler_file, filler_type in [
        ("fillers_pt_br.json", "pt_br"),
        ("fillers_fr_pt.json", "fr_pt"),
    ]:
        fpath = labeled_dir / filler_file
        if fpath.exists():
            with open(fpath) as f:
                fillers = json.load(f)
            for fl in fillers:
                if fl.get("with_filler"):
                    all_sentences.append({
                        "text": fl["with_filler"],
                        "label": "incomplete",
                        "source": f"filler_{filler_type}",
                    })
            log.info("Loaded %d %s filler sentences", len(fillers), filler_type)

    # 3. French-Portuguese sentences
    frpt_path = labeled_dir / "french_portuguese.json"
    if frpt_path.exists():
        with open(frpt_path) as f:
            frpt = json.load(f)
        for s in frpt:
            label = "complete" if s.get("label", "") == "completo" else "incomplete"
            all_sentences.append({
                "text": s["text"],
                "label": label,
                "source": "claude_fr_pt",
            })
        log.info("Loaded %d French-Portuguese sentences", len(frpt))

    if not all_sentences:
        log.error("No labeled sentences found in %s — run 02_generate_labels.py first", labeled_dir)
        return DATA_DIR

    log.info("Total sentences to synthesize: %d", len(all_sentences))

    # Split: native PT-BR vs French-accented
    native_sentences = [s for s in all_sentences if s["source"] != "claude_fr_pt"]
    french_sentences = [s for s in all_sentences if s["source"] == "claude_fr_pt"]

    # Also use some filler_fr_pt sentences with XTTS
    fr_filler = [s for s in all_sentences if s["source"] == "filler_fr_pt"]
    french_sentences.extend(fr_filler)

    random.shuffle(native_sentences)
    random.shuffle(french_sentences)
    native_sentences = native_sentences[:max_native_sentences]
    french_sentences = french_sentences[:max_french_sentences]

    # Generate audio
    log.info("\n=== Generating native PT-BR audio (Kokoro) ===")
    native_samples = generate_kokoro_audio(native_sentences)

    log.info("\n=== Generating French-accented audio (XTTS) ===")
    ref_dir = DATA_DIR / "french_reference_audio"
    french_samples = generate_xtts_audio(french_sentences, reference_audio_dir=ref_dir)

    # NEW: Short utterances (Pipecat v3.2: -40% errors)
    log.info("\n=== Generating short utterance dataset ===")
    short_samples = generate_short_utterances(n_per_utterance=3)

    # NEW: Hesitation pause variants (SpeculativeETD V3: best data variant)
    # Critical for French speakers: long pauses mid-sentence are NOT end-of-turn
    log.info("\n=== Creating hesitation-pause variants ===")
    all_pre_hesitation = native_samples + french_samples + short_samples
    hesitation_variants = create_hesitation_variants(all_pre_hesitation, fraction=0.3)

    all_samples = native_samples + french_samples + short_samples + hesitation_variants
    random.shuffle(all_samples)

    log.info("Total samples before augmentation: %d", len(all_samples))
    log.info("  Native PT-BR: %d", len(native_samples))
    log.info("  French-accented: %d", len(french_samples))
    log.info("  Short utterances: %d", len(short_samples))
    log.info("  Hesitation variants: %d", len(hesitation_variants))

    # Save
    log.info("\n=== Saving dataset ===")
    output_dir = DATA_DIR / "tts_dataset"
    save_dataset(all_samples, output_dir, augment_copies=augment_copies)

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    random.seed(42)
    np.random.seed(42)

    run_audio_generation(
        max_native_sentences=3000,
        max_french_sentences=1000,
        augment_copies=2,
    )
