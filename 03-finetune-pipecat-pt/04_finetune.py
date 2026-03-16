"""Fine-tune Pipecat Smart Turn v3 for Portuguese + French-accented Portuguese.

Architecture: exact SmartTurnV3Model from Pipecat's train.py
- WhisperEncoder (openai/whisper-tiny) with max_source_positions=400 (8s)
- Attention pooling: Linear(384→256) → Tanh → Linear(256→1)
- Classifier: Linear(384→256) → LayerNorm → GELU → Dropout(0.1)
                → Linear(256→64) → GELU → Linear(64→1)

Training strategy:
- Initialize from openai/whisper-tiny (no Pipecat PyTorch weights available)
- Use Pipecat's Portuguese data as primary training data
- Add our Claude-labeled + TTS-generated data for PT-BR + French accent
- Pipecat hyperparams: lr=5e-5, warmup=0.2, cosine schedule, epochs=4

Changes vs initial plan (based on reference analysis — see references/):
- alpha=0.25 (not 0.6) per Lin et al. 2017 optimal for gamma=2
- batch_size=128 (not 32) per Pipecat train.py (they use 384)
- Removed label smoothing (double-regularization with focal loss, per EMNLP 2022)
- Added BCEWithLogitsLoss option for comparison (Pipecat's original loss)
- Added real noise augmentation (not just Gaussian) per Pipecat v3.2
- Added short utterance dataset loading per Pipecat v3.2 findings
- ONNX opset 18 (not 17) per Pipecat train.py
- Added INT8 static quantization step per Pipecat's deployment pipeline

Language-learning-specific improvements (March 2026 research):
- fp_penalty=2.0: asymmetric cost — interrupting a learner costs 2x more than
  waiting too long (ConversAR 2025, Praktika approach)
- Speak & Improve L2 corpus: real L2 learner speech with disfluencies (340h)
- Dual threshold evaluation: eager (speculative) + final (confirmed) per Deepgram Flux

Data sources:
1. Pipecat v3.2 Portuguese subset (~5K samples from 270K)
2. Our custom TTS dataset (03_generate_audio.py output)
3. CORAA real audio (01_download_pipecat.py output)
4. Speak & Improve L2 corpus (~2K samples, cross-lingual hesitation patterns)
"""

from __future__ import annotations

import gc
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE

_workspace = Path("/workspace") if Path("/workspace").exists() else Path(".")
OUTPUT_DIR = _workspace / "results"
CACHE_DIR = _workspace / "hf_cache"
DATA_DIR = Path(__file__).parent / "data"

# Label smoothing REMOVED — focal loss already provides implicit calibration
# via entropy regularization (EMNLP 2022: focal_loss_calibration_emnlp_2022.md)
# Double-regularization (FL + LS) reduces discriminative power.
LABEL_SMOOTH = 0.0


# ---------------------------------------------------------------------------
# Model — SmartTurnV3Model (exact Pipecat architecture)
# ---------------------------------------------------------------------------

class SmartTurnV3Model(nn.Module):
    """Pipecat Smart Turn v3 model architecture.

    Matches the exact architecture from:
    https://github.com/pipecat-ai/smart-turn/blob/main/train/train.py

    WhisperEncoder (whisper-tiny, 384-dim) → attention pooling → classifier
    """

    def __init__(self, whisper_model: str = "openai/whisper-tiny"):
        super().__init__()
        from transformers import WhisperModel

        # Load pretrained encoder
        whisper = WhisperModel.from_pretrained(
            whisper_model, cache_dir=str(CACHE_DIR)
        )
        self.encoder = whisper.encoder

        # Resize position embeddings: 1500 (30s) → 400 (8s)
        max_pos = 400
        old_embed = self.encoder.embed_positions.weight.data
        new_embed = old_embed[:max_pos, :]
        self.encoder.embed_positions = nn.Embedding(max_pos, old_embed.shape[1])
        self.encoder.embed_positions.weight.data = new_embed
        self.encoder.config.max_source_positions = max_pos

        hidden_size = self.encoder.config.d_model  # 384 for whisper-tiny

        # Attention pooling (exact Pipecat architecture)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Classifier head (exact Pipecat architecture)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input_features).last_hidden_state
        attn_weights = self.attention(encoder_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (encoder_output * attn_weights).sum(dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) — focuses on hard boundary cases.

    alpha=0.25 is optimal for gamma=2.0 per the original paper (Table 1a).
    Our initial alpha=0.6 over-weighted positives and hurt calibration.

    fp_penalty: extra multiplier on false-positive loss (model says "complete"
    when speaker is still talking). For a language-learning avatar, interrupting
    the learner mid-thought is much worse than waiting too long.
    - ConversAR (2025) gives learners "infinite thinking period"
    - Praktika uses extended silence tolerance for L2 speech
    - Default 2.0 means FP errors cost 2x more than FN errors
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, fp_penalty: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.fp_penalty = fp_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce

        # Asymmetric cost: penalize FP (interrupting learner) more than FN (waiting)
        # FP = model predicts 1 (complete) when target is 0 (incomplete)
        if self.fp_penalty != 1.0:
            is_fp = (probs > 0.5).float() * (1 - targets)  # predicted complete, actually incomplete
            penalty = 1.0 + (self.fp_penalty - 1.0) * is_fp
            loss = loss * penalty

        return loss.mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class AudioSample:
    audio: np.ndarray
    label: float  # 1.0 = complete, 0.0 = incomplete
    source: str
    speaker_id: str = ""


def load_pipecat_portuguese(max_samples: int = 5000) -> list[AudioSample]:
    """Load Pipecat v3.2 Portuguese data (from 01_download_pipecat.py)."""
    audio_dir = DATA_DIR / "pipecat_pt_audio"
    if not audio_dir.exists():
        log.warning("Pipecat PT audio not found at %s — run 01_download_pipecat.py first", audio_dir)
        return []

    import soundfile as sf

    samples = []
    for wav_path in sorted(audio_dir.glob("*.wav"))[:max_samples]:
        try:
            audio, sr = sf.read(str(wav_path))
            audio = np.array(audio, dtype=np.float32)

            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

            label = 1.0 if "_complete" in wav_path.name else 0.0
            audio = _extract_window(audio)
            if audio is not None:
                samples.append(AudioSample(
                    audio=audio, label=label,
                    source="pipecat_v3.2",
                    speaker_id=f"pipecat_{wav_path.stem.split('_')[0]}",
                ))
        except Exception as e:
            log.warning("Error loading %s: %s", wav_path.name, e)

    n_c = sum(1 for s in samples if s.label == 1.0)
    n_i = sum(1 for s in samples if s.label == 0.0)
    log.info("Pipecat PT: %d samples (%d complete, %d incomplete)", len(samples), n_c, n_i)
    return samples


def load_tts_dataset(max_samples: int = 10000) -> list[AudioSample]:
    """Load TTS-generated audio (from 03_generate_audio.py)."""
    tts_dir = DATA_DIR / "tts_dataset"
    meta_path = tts_dir / "metadata.json"

    if not meta_path.exists():
        log.warning("TTS dataset not found at %s — run 03_generate_audio.py first", tts_dir)
        return []

    import soundfile as sf

    with open(meta_path) as f:
        metadata = json.load(f)

    audio_dir = tts_dir / "audio"
    samples = []

    for meta in metadata[:max_samples]:
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

            label = 1.0 if meta["label"] == "complete" else 0.0
            audio = _extract_window(audio)
            if audio is not None:
                samples.append(AudioSample(
                    audio=audio, label=label,
                    source=f"tts_{meta['accent']}",
                    speaker_id=meta["voice"],
                ))
        except Exception as e:
            log.warning("Error loading %s: %s", meta["file"], e)

    n_c = sum(1 for s in samples if s.label == 1.0)
    n_i = sum(1 for s in samples if s.label == 0.0)
    log.info("TTS dataset: %d samples (%d complete, %d incomplete)", len(samples), n_c, n_i)
    return samples


def load_coraa_real_audio(max_samples: int = 3000) -> list[AudioSample]:
    """Load REAL conversational audio from CORAA (not TTS).

    SpeculativeETD showed synthetic-only training has a devastating gap:
    F1 drops from 94.7% → 30.3% on real data. Mixing real audio is critical.

    CORAA MUPE has 365h of real PT-BR interviews with speaker diarization.
    We use these as COMPLETE samples (full utterances with natural prosody).
    For INCOMPLETE, we truncate at natural pause points.
    """
    from datasets import load_dataset
    import re

    log.info("Loading CORAA MUPE real audio (streaming)...")
    try:
        ds = load_dataset(
            "nilc-nlp/CORAA-MUPE-ASR",
            split="train",
            streaming=True,
            cache_dir=str(CACHE_DIR),
        )
    except Exception as e:
        log.warning("Failed to load CORAA MUPE: %s", e)
        return []

    samples = []
    complete_count = 0
    incomplete_count = 0
    target_per_class = max_samples // 2

    for i, row in enumerate(ds):
        if complete_count >= target_per_class and incomplete_count >= target_per_class:
            break

        try:
            audio_data = row.get("audio", {})
            if not audio_data:
                continue

            audio = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]

            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

            duration = len(audio) / SAMPLE_RATE
            if duration < 1.0:
                continue

            text = str(row.get("text", ""))
            speaker_id = f"coraa_real_{i // 50}"

            # COMPLETE: full utterances ending with sentence punctuation
            if complete_count < target_per_class and re.search(r'[.!?]+\s*$', text):
                window = _extract_window(audio)
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        source="coraa_real",
                        speaker_id=speaker_id,
                    ))
                    complete_count += 1

            # INCOMPLETE: truncate real audio at 40-70% (natural mid-utterance)
            if incomplete_count < target_per_class and duration >= 2.0:
                cut_frac = random.uniform(0.4, 0.7)
                truncated = audio[:int(len(audio) * cut_frac)]
                window = _extract_window(truncated)
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        source="coraa_real",
                        speaker_id=speaker_id,
                    ))
                    incomplete_count += 1

        except Exception:
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  CORAA real: scanned %d, %d complete, %d incomplete", i, complete_count, incomplete_count)

    log.info("CORAA real audio: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
    return samples


def load_pipecat_test_data() -> list[AudioSample]:
    """Load Pipecat v3.2 Portuguese test data for evaluation."""
    test_dir = DATA_DIR / "pipecat_pt_test"
    if not test_dir.exists():
        log.warning("Pipecat PT test data not found — run 01_download_pipecat.py first")
        return []

    import soundfile as sf

    samples = []
    for wav_path in sorted(test_dir.glob("*.wav")):
        try:
            audio, sr = sf.read(str(wav_path))
            audio = np.array(audio, dtype=np.float32)

            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

            label = 1.0 if "_complete" in wav_path.name else 0.0
            audio = _extract_window(audio)
            if audio is not None:
                samples.append(AudioSample(
                    audio=audio, label=label,
                    source="pipecat_test",
                    speaker_id=f"pipecat_test_{wav_path.stem.split('_')[0]}",
                ))
        except Exception as e:
            log.warning("Error loading test %s: %s", wav_path.name, e)

    log.info("Pipecat test: %d samples", len(samples))
    return samples


def _extract_window(audio: np.ndarray) -> np.ndarray | None:
    """Extract 8-second window from end of audio, pad if needed."""
    if len(audio) < SAMPLE_RATE:  # minimum 1s
        return None

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
    elif len(audio) < WINDOW_SAMPLES:
        padding = WINDOW_SAMPLES - len(audio)
        audio = np.pad(audio, (padding, 0), mode="constant")

    # ~200ms silence at end (VAD behavior)
    silence_samples = int(0.2 * SAMPLE_RATE)
    audio[-silence_samples:] = 0.0

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------------------------

# Cache for background noise samples (loaded once)
_noise_cache: list[np.ndarray] = []


def _load_noise_samples() -> list[np.ndarray]:
    """Load real background noise samples for augmentation.

    Pipecat v3.2 used CC-0 Freesound.org cafe/office noise and saw
    40% fewer short-utterance misclassifications (pipecat_smart_turn_v3_2.md).
    """
    global _noise_cache
    if _noise_cache:
        return _noise_cache

    noise_dir = DATA_DIR / "noise_samples"
    if not noise_dir.exists():
        return []

    import soundfile as sf
    for wav_path in noise_dir.glob("*.wav"):
        try:
            audio, sr = sf.read(str(wav_path))
            audio = np.array(audio, dtype=np.float32)
            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()
            _noise_cache.append(audio)
        except Exception:
            pass

    if _noise_cache:
        log.info("Loaded %d background noise samples for augmentation", len(_noise_cache))
    return _noise_cache


def augment_audio(audio: np.ndarray) -> np.ndarray:
    """Data augmentation for training.

    Per reference analysis:
    - Real background noise (Pipecat v3.2: 40% fewer errors with cafe/office noise)
    - Speed perturbation (standard in speech ML)
    - Volume variation (simulates mic distance)
    - Time shift
    """
    aug = audio.copy()

    # Speed perturbation
    if random.random() < 0.5:
        speed = random.uniform(0.9, 1.1)
        indices = np.arange(0, len(aug), speed).astype(int)
        indices = indices[indices < len(aug)]
        aug = aug[indices]
        if len(aug) > WINDOW_SAMPLES:
            aug = aug[:WINDOW_SAMPLES]
        elif len(aug) < WINDOW_SAMPLES:
            aug = np.pad(aug, (WINDOW_SAMPLES - len(aug), 0), mode="constant")

    # Volume variation
    if random.random() < 0.5:
        aug *= random.uniform(0.6, 1.4)

    # Real background noise (preferred) or Gaussian fallback
    noise_samples = _load_noise_samples()
    if noise_samples and random.random() < 0.5:
        noise = random.choice(noise_samples)
        # Loop or trim noise to match audio length
        if len(noise) < len(aug):
            repeats = len(aug) // len(noise) + 1
            noise = np.tile(noise, repeats)[:len(aug)]
        else:
            start = random.randint(0, len(noise) - len(aug))
            noise = noise[start:start + len(aug)]
        snr_db = random.uniform(10, 25)  # 10-25 dB SNR
        noise_scale = np.sqrt(np.mean(aug ** 2)) / (np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20) + 1e-8)
        aug += noise * noise_scale
    elif random.random() < 0.4:
        # Gaussian noise fallback
        noise_level = random.uniform(0.002, 0.02)
        aug += np.random.randn(len(aug)).astype(np.float32) * noise_level

    # Time shift
    if random.random() < 0.3:
        shift = random.randint(-int(0.3 * SAMPLE_RATE), int(0.3 * SAMPLE_RATE))
        aug = np.roll(aug, shift)
        if shift > 0:
            aug[:shift] = 0.0
        elif shift < 0:
            aug[shift:] = 0.0

    return np.clip(aug, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SmartTurnDataset(Dataset):
    def __init__(
        self,
        samples: list[AudioSample],
        feature_extractor: WhisperFeatureExtractor,
        augment: bool = False,
    ):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio = sample.audio

        if self.augment:
            audio = augment_audio(audio)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )

        features = inputs.input_features.squeeze(0).astype(np.float32)

        return {
            "input_features": torch.from_numpy(features),
            "labels": torch.tensor(sample.label, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_by_speaker(
    samples: list[AudioSample],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[AudioSample], list[AudioSample], list[AudioSample]]:
    """Split by speaker ID to prevent data leakage."""
    speaker_map: dict[str, list[AudioSample]] = {}
    for s in samples:
        speaker_map.setdefault(s.speaker_id or "unknown", []).append(s)

    speakers = list(speaker_map.keys())
    random.shuffle(speakers)

    n_val = max(1, int(len(speakers) * val_frac))
    n_test = max(1, int(len(speakers) * test_frac))

    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test:n_test + n_val])
    train_spk = set(speakers[n_test + n_val:])

    train = [s for sp in train_spk for s in speaker_map[sp]]
    val = [s for sp in val_spk for s in speaker_map[sp]]
    test = [s for sp in test_spk for s in speaker_map[sp]]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_c = sum(1 for s in split if s.label == 1.0)
        n_i = sum(1 for s in split if s.label == 0.0)
        log.info("  %s: %d samples (%d complete, %d incomplete, %d speakers)",
                 name, len(split), n_c, n_i, len(set(s.speaker_id for s in split)))

    return train, val, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_speak_improve_l2(max_samples: int = 2000) -> list[AudioSample]:
    """Load L2 learner speech from Speak & Improve Corpus 2025.

    340 hours of L2 English learner speech with disfluency annotations and
    CEFR proficiency scores (A2-C1, majority B1-B2).

    Even though this is English L2 (not Portuguese), L2 hesitation patterns
    transfer across languages (Cenoz 2000). The pauses, false starts, and
    disfluencies of L2 speakers share universal characteristics regardless
    of target language.

    Knill et al. (2025). Speak & Improve Corpus 2025: an L2 English Speech
    Corpus for Language Assessment and Feedback. arXiv:2412.11986
    """
    from datasets import load_dataset

    log.info("Loading Speak & Improve L2 corpus (streaming)...")
    try:
        ds = load_dataset(
            "CambridgeEnglish/SpeakAndImprove2025",
            split="train",
            streaming=True,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning("Failed to load Speak & Improve corpus: %s — trying fallback name", e)
        try:
            ds = load_dataset(
                "cambridgeenglishtests/speak-and-improve-2025",
                split="train",
                streaming=True,
                cache_dir=str(CACHE_DIR),
                trust_remote_code=True,
            )
        except Exception as e2:
            log.warning("Speak & Improve corpus not available: %s — skipping", e2)
            return []

    samples = []
    complete_count = 0
    incomplete_count = 0
    target_per_class = max_samples // 2

    for i, row in enumerate(ds):
        if complete_count >= target_per_class and incomplete_count >= target_per_class:
            break

        try:
            audio_data = row.get("audio", {})
            if not audio_data:
                continue

            audio = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]

            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

            duration = len(audio) / SAMPLE_RATE
            if duration < 1.0:
                continue

            text = str(row.get("text", row.get("transcript", "")))
            speaker_id = f"si_l2_{i // 20}"

            # COMPLETE: full utterances (use entire audio)
            if complete_count < target_per_class and duration >= 2.0:
                window = _extract_window(audio)
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        source="speak_improve_l2",
                        speaker_id=speaker_id,
                    ))
                    complete_count += 1

            # INCOMPLETE: truncate at random point (simulates mid-utterance)
            if incomplete_count < target_per_class and duration >= 3.0:
                cut_frac = random.uniform(0.3, 0.6)
                truncated = audio[:int(len(audio) * cut_frac)]
                # Add hesitation-like pause at end (L2 speakers pause 524ms+ per Kosmala 2022)
                pause_ms = random.uniform(500, 2000)
                pause_samples = int(pause_ms / 1000 * SAMPLE_RATE)
                pause = np.random.randn(pause_samples).astype(np.float32) * 0.001
                truncated = np.concatenate([truncated, pause])
                window = _extract_window(truncated)
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        source="speak_improve_l2",
                        speaker_id=speaker_id,
                    ))
                    incomplete_count += 1

        except Exception:
            continue

        if i % 2000 == 0 and i > 0:
            log.info("  S&I L2: scanned %d, %d complete, %d incomplete", i, complete_count, incomplete_count)

    log.info("Speak & Improve L2: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
    return samples


def train(
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 5e-5,
    warmup_ratio: float = 0.2,
    max_pipecat_samples: int = 5000,
    max_tts_samples: int = 10000,
    max_l2_samples: int = 2000,
    whisper_model: str = "openai/whisper-tiny",
    loss_fn: str = "focal",  # "focal" or "bce" (Pipecat's original)
    fp_penalty: float = 2.0,  # asymmetric cost: FP costs 2x more than FN
) -> Path:
    """Fine-tune SmartTurnV3Model on Portuguese data.

    Default hyperparams from Pipecat's train.py:
    - lr=5e-5, warmup_ratio=0.2, cosine schedule
    - Pipecat uses epochs=4 on 270K samples; we use 6 on 5-15K (more passes needed)
    - Pipecat uses batch_size=384; we use 128 (A10G has 24GB, enough for 128)
    - loss_fn="focal" (our improvement) or "bce" (Pipecat's original) for comparison

    Language-learning-specific improvements (March 2026 research):
    - fp_penalty=2.0: asymmetric cost — interrupting a learner mid-thought is
      much worse than waiting too long (ConversAR 2025, Praktika approach)
    - Speak & Improve L2 corpus: real L2 learner speech with hesitations/disfluencies
    - Dual threshold in evaluation: eager (0.3-0.5) for speculative prep,
      final (0.7+) for actual turn transition (inspired by Deepgram Flux)
    """

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Training on device: %s", device)
    if device == "cuda":
        log.info("GPU: %s (%d MB)", torch.cuda.get_device_name(),
                 torch.cuda.get_device_properties(0).total_memory // 1024 // 1024)

    # ----- Load all data sources -----
    t0 = time.time()
    all_samples: list[AudioSample] = []

    # 1. Pipecat Portuguese (primary — these have LLM-curated labels)
    pipecat = load_pipecat_portuguese(max_samples=max_pipecat_samples)
    all_samples.extend(pipecat)
    del pipecat
    gc.collect()

    # 2. Our TTS-generated data (native PT-BR + French accent + short utterances)
    tts = load_tts_dataset(max_samples=max_tts_samples)
    all_samples.extend(tts)
    del tts
    gc.collect()

    # 3. CORAA real audio (critical: SpeculativeETD showed synthetic→real gap of 94.7%→30.3%)
    coraa = load_coraa_real_audio(max_samples=3000)
    all_samples.extend(coraa)
    del coraa
    gc.collect()

    # 4. Speak & Improve L2 corpus — real L2 learner speech with disfluencies
    # L2 hesitation patterns transfer across languages (Cenoz 2000)
    # 340h of L2 English learners, CEFR A2-C1 (arXiv:2412.11986)
    si_l2 = load_speak_improve_l2(max_samples=max_l2_samples)
    all_samples.extend(si_l2)
    del si_l2
    gc.collect()

    if not all_samples:
        raise RuntimeError("No training samples loaded! Run 01/03 scripts first.")

    load_time = time.time() - t0
    n_c = sum(1 for s in all_samples if s.label == 1.0)
    n_i = sum(1 for s in all_samples if s.label == 0.0)
    sources = {}
    for s in all_samples:
        sources[s.source] = sources.get(s.source, 0) + 1

    log.info("Total: %d samples (%d complete, %d incomplete) in %.0fs", len(all_samples), n_c, n_i, load_time)
    for src, cnt in sorted(sources.items()):
        log.info("  %s: %d", src, cnt)

    # ----- Split -----
    log.info("=== Splitting by speaker ===")
    train_samples, val_samples, internal_test = split_by_speaker(all_samples)

    # Also load Pipecat test data (held-out, never seen during training)
    pipecat_test = load_pipecat_test_data()

    # ----- Datasets -----
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    train_ds = SmartTurnDataset(train_samples, feature_extractor, augment=True)
    val_ds = SmartTurnDataset(val_samples, feature_extractor, augment=False)

    # Balanced sampling
    train_labels = [s.label for s in train_samples]
    n_pos = sum(1 for l in train_labels if l == 1.0)
    n_neg = len(train_labels) - n_pos
    weights = [1.0 / n_neg if l == 0.0 else 1.0 / n_pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    use_pin = device == "cuda"
    n_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=n_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_pin)

    # ----- Model -----
    model = SmartTurnV3Model(whisper_model=whisper_model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d params (%.1f MB)", total_params, total_params * 4 / 1024 / 1024)

    # ----- Loss -----
    # Pipecat uses BCEWithLogitsLoss with dynamic pos_weight (clamped 0.1-10.0)
    # We default to Focal Loss (alpha=0.25, gamma=2.0 per Lin et al. 2017)
    # but support BCE for comparison
    pos_weight_val = min(max(n_neg / max(n_pos, 1), 0.1), 10.0)
    pos_weight = torch.tensor([pos_weight_val], device=device)

    if loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log.info("Loss: BCEWithLogitsLoss (Pipecat original), pos_weight=%.2f", pos_weight_val)
    else:
        criterion = FocalLoss(gamma=2.0, alpha=0.25, fp_penalty=fp_penalty)
        log.info("Loss: FocalLoss (gamma=2.0, alpha=0.25, fp_penalty=%.1f)", fp_penalty)
        log.info("  Asymmetric cost: FP (interrupting learner) penalized %.1fx more than FN", fp_penalty)

    # ----- Optimizer: uniform LR (Pipecat uses single lr=5e-5 for all params) -----
    # Pipecat's train.py does NOT use differential LR — all params at same rate.
    # Since we initialize from whisper-tiny (same as Pipecat), uniform LR is correct.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule with warmup (Pipecat: warmup_ratio=0.2)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Training loop -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_path = OUTPUT_DIR / "best_model.pt"
    resume_path = OUTPUT_DIR / "resume_checkpoint.pt"
    patience = 5
    patience_counter = 0
    history = []
    start_epoch = 0

    # Resume support
    if resume_path.exists():
        log.info("=== Resuming from checkpoint ===")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_f1 = ckpt.get("best_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", [])
        log.info("  Resumed at epoch %d, best_f1=%.4f", start_epoch, best_f1)

    log.info("=== Training: %d epochs, batch=%d, lr=%.1e, warmup_ratio=%.1f ===",
             epochs, batch_size, lr, warmup_ratio)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            hard_labels = (labels > 0.5).float()
            train_correct += (preds == hard_labels).sum().item()
            train_total += len(labels)

            if batch_idx % 50 == 0 and batch_idx > 0:
                log.info("  batch %d/%d loss=%.4f", batch_idx, len(train_loader), loss.item())

        # Validate
        model.eval()
        val_metrics = _evaluate(model, val_loader, device, criterion)
        train_acc = train_correct / max(train_total, 1)
        epoch_time = time.time() - t_epoch

        log.info(
            "Epoch %d/%d (%.0fs): train_loss=%.4f train_acc=%.3f | "
            "val_acc=%.3f val_f1=%.3f prec=%.3f rec=%.3f",
            epoch + 1, epochs, epoch_time,
            train_loss / max(train_total, 1), train_acc,
            val_metrics["accuracy"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_f1": best_f1,
                "val_metrics": val_metrics,
                "whisper_model": whisper_model,
            }, best_path)
            log.info("  -> New best model (val_f1=%.4f)", best_f1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        # Resume checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "history": history,
        }, resume_path)

    if resume_path.exists():
        resume_path.unlink()

    # ----- Final evaluation on internal test split -----
    log.info("\n=== Internal Test Evaluation ===")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if internal_test:
        test_ds = SmartTurnDataset(internal_test, feature_extractor, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        test_metrics = _evaluate(model, test_loader, device, criterion)
        _log_metrics("Internal test", test_metrics)
    else:
        test_metrics = {}

    # ----- Pipecat test set (baseline comparison) -----
    pipecat_test_metrics = {}
    if pipecat_test:
        log.info("\n=== Pipecat PT Test Evaluation (baseline comparison) ===")
        pipecat_ds = SmartTurnDataset(pipecat_test, feature_extractor, augment=False)
        pipecat_loader = DataLoader(pipecat_ds, batch_size=batch_size, shuffle=False)
        pipecat_test_metrics = _evaluate(model, pipecat_loader, device, criterion)
        _log_metrics("Pipecat PT test", pipecat_test_metrics)
        log.info("  Pipecat baseline: 95.42%% accuracy, 2.79%% FP, 1.79%% FN")

    # ----- Threshold sweep -----
    log.info("\n=== Threshold Sweep ===")
    threshold_results = {}
    eval_loader = test_loader if internal_test else val_loader
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        t_metrics = _evaluate(model, eval_loader, device, criterion, threshold=thresh)
        threshold_results[str(thresh)] = t_metrics
        log.info("  threshold=%.2f: prec=%.3f rec=%.3f f1=%.3f acc=%.3f FP=%.3f",
                 thresh, t_metrics["precision"], t_metrics["recall"],
                 t_metrics["f1"], t_metrics["accuracy"], t_metrics["fp_rate"])

    # ----- Dual threshold recommendation (inspired by Deepgram Flux) -----
    # For a language-learning avatar:
    # - eager_threshold (0.3-0.5): start preparing response speculatively
    #   (e.g., begin LLM generation) — reduces perceived latency
    # - final_threshold (0.7+): actually take the turn and speak
    #   Higher final threshold = fewer interruptions (critical for L2 learners)
    log.info("\n=== Dual Threshold Recommendation (Deepgram Flux-inspired) ===")
    # Find final threshold: maximize precision with recall >= 85%
    # (we want very few interruptions, even at the cost of some missed turns)
    final_candidates = [(k, v) for k, v in threshold_results.items()
                        if v["recall"] >= 0.85 and float(k) >= 0.6]
    if final_candidates:
        best_final = min(final_candidates, key=lambda x: x[1]["fp_rate"])
        log.info("  Recommended final_threshold: %.2f (FP=%.1f%%, prec=%.3f, rec=%.3f)",
                 float(best_final[0]), best_final[1]["fp_rate"] * 100,
                 best_final[1]["precision"], best_final[1]["recall"])
    else:
        best_final = ("0.7", threshold_results.get("0.7", {}))
        log.info("  Default final_threshold: 0.70")

    # Eager threshold: lower confidence where we start speculative processing
    eager_thresh = max(0.3, float(best_final[0]) - 0.3)
    log.info("  Recommended eager_threshold: %.2f (start LLM generation speculatively)",
             eager_thresh)
    log.info("  Latency savings: ~150-250ms earlier response start (Deepgram Flux benchmark)")

    # ----- ONNX export (FP32 → INT8, per Pipecat's deploy pipeline) -----
    model = model.to("cpu")
    onnx_fp32_path = OUTPUT_DIR / "smart_turn_pt_v3_fp32.onnx"
    onnx_int8_path = OUTPUT_DIR / "smart_turn_pt_v3.onnx"
    dummy = torch.randn(1, 80, 800)
    try:
        # Step 1: Export FP32 ONNX (opset 18, per Pipecat train.py)
        torch.onnx.export(
            model, dummy, str(onnx_fp32_path),
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )
        log.info("ONNX FP32 exported: %s (%.1f MB)", onnx_fp32_path,
                 onnx_fp32_path.stat().st_size / 1024 / 1024)

        # Step 2: INT8 static quantization (entropy calibration, per Pipecat)
        # Pipecat uses: 1024 calibration samples, QDQ format, Entropy method
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat

            class SmartTurnCalibrationReader(CalibrationDataReader):
                def __init__(self, dataset, n_samples=1024):
                    self.data = []
                    for i, sample in enumerate(dataset):
                        if i >= n_samples:
                            break
                        self.data.append({"input_features": sample["input_features"].unsqueeze(0).numpy()})
                    self.idx = 0

                def get_next(self):
                    if self.idx >= len(self.data):
                        return None
                    result = self.data[self.idx]
                    self.idx += 1
                    return result

            # Use validation data for calibration
            calib_reader = SmartTurnCalibrationReader(val_ds, n_samples=1024)
            quantize_static(
                str(onnx_fp32_path),
                str(onnx_int8_path),
                calib_reader,
                quant_format=QuantFormat.QDQ,
            )
            log.info("ONNX INT8 exported: %s (%.1f MB)", onnx_int8_path,
                     onnx_int8_path.stat().st_size / 1024 / 1024)
        except ImportError:
            log.warning("onnxruntime.quantization not available — saving FP32 only")
            import shutil
            shutil.copy2(onnx_fp32_path, onnx_int8_path)
        except Exception as e:
            log.warning("INT8 quantization failed: %s — saving FP32 only", e)
            import shutil
            shutil.copy2(onnx_fp32_path, onnx_int8_path)

    except Exception as e:
        log.warning("ONNX export failed: %s", e)

    # ----- Save results -----
    results = {
        "model": "smart_turn_pt_v3_finetuned",
        "whisper_model": whisper_model,
        "architecture": "SmartTurnV3Model (exact Pipecat)",
        "target_domain": "language_learning_avatar",
        "learner_profile": "francophone_learning_portuguese",
        "total_samples": len(all_samples),
        "sources": sources,
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "pipecat_test_metrics": pipecat_test_metrics,
        "pipecat_baseline": {"accuracy": 0.9542, "fp_rate": 0.0279, "fn_rate": 0.0179},
        "threshold_sweep": threshold_results,
        "dual_threshold": {
            "eager_threshold": eager_thresh,
            "final_threshold": float(best_final[0]),
            "rationale": "For L2 language learning: higher final threshold reduces "
                        "interruptions. Eager threshold enables speculative LLM "
                        "generation 150-250ms earlier (Deepgram Flux approach).",
        },
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "label_smoothing": LABEL_SMOOTH,
            "focal_loss_gamma": 2.0,
            "focal_loss_alpha": 0.25,
            "fp_penalty": fp_penalty,
            "loss_fn": loss_fn,
        },
    }

    results_path = OUTPUT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)

    return onnx_int8_path


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model at a given threshold."""
    correct = total = tp = fp = fn = tn = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = (torch.sigmoid(logits) > threshold).float()
            hard_labels = (labels > 0.5).float()
            correct += (preds == hard_labels).sum().item()
            total += len(labels)
            tp += ((preds == 1) & (hard_labels == 1)).sum().item()
            fp += ((preds == 1) & (hard_labels == 0)).sum().item()
            fn += ((preds == 0) & (hard_labels == 1)).sum().item()
            tn += ((preds == 0) & (hard_labels == 0)).sum().item()

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "loss": round(total_loss / max(total, 1), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fp_rate": round(fp / max(fp + tn, 1), 4),
        "fn_rate": round(fn / max(fn + tp, 1), 4),
    }


def _log_metrics(name: str, metrics: dict) -> None:
    log.info("%s results:", name)
    log.info("  Accuracy:  %.3f", metrics["accuracy"])
    log.info("  Precision: %.3f", metrics["precision"])
    log.info("  Recall:    %.3f", metrics["recall"])
    log.info("  F1:        %.3f", metrics["f1"])
    log.info("  FP rate:   %.3f", metrics["fp_rate"])
    log.info("  FN rate:   %.3f", metrics["fn_rate"])
    log.info("  TP=%d FP=%d FN=%d TN=%d", metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train(
        epochs=6,
        batch_size=128,
        lr=5e-5,
        warmup_ratio=0.2,
        max_pipecat_samples=5000,
        max_tts_samples=10000,
        max_l2_samples=2000,
        whisper_model="openai/whisper-tiny",
        loss_fn="focal",  # or "bce" for Pipecat's original
        fp_penalty=2.0,   # interrupting learner costs 2x more than waiting
    )
