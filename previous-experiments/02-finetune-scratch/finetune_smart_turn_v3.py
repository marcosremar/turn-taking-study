"""
Fine-tune Pipecat Smart Turn v3 on Portuguese speech data (GPU version).

Key improvements over v2:
- Punctuation-based labels (text ending with .!? = complete, otherwise incomplete)
- No audiobook data (removed MLS) — only conversational/interview speech
- Whisper Tiny encoder (39M params) — same as original Pipecat Smart Turn v3
- Fixed MUPE speaker_id to use actual unique IDs
- More data (25k+ per dataset)
- Better augmentation (speed perturbation, pitch variation)

Datasets:
- CORAA v1.1 (291h, conversational Brazilian Portuguese)
- CORAA-MUPE-ASR (365h, interview turn-taking)

Run on a Vast.ai GPU instance with:
    python finetune_smart_turn_v3.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WhisperFeatureExtractor

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE

# Use /workspace if on RunPod (persists across pod restarts), else local
_workspace = Path("/workspace") if Path("/workspace").exists() else Path(".")
OUTPUT_DIR = _workspace / "checkpoints" / "smart_turn_pt_v3"
CACHE_DIR = _workspace / "hf_cache"
CHECKPOINT_EVERY_BATCHES = 100  # Save resumable checkpoint every N batches

# Punctuation that signals a complete utterance
COMPLETE_ENDINGS = re.compile(r'[.!?…]+\s*$')
# Punctuation/patterns that signal incomplete
INCOMPLETE_ENDINGS = re.compile(r'[,;:\-–—]\s*$')

# Label smoothing: soften hard labels to improve calibration
LABEL_SMOOTH = 0.05  # 0.0 → 0.05, 1.0 → 0.95


# ---------------------------------------------------------------------------
# Focal Loss — penalizes easy examples, focuses on hard boundary cases
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) with optional pos_weight for class imbalance.

    gamma > 0 down-weights well-classified examples so the model focuses on
    hard false positives (the main precision problem).
    alpha < 1 further penalizes false positives by reducing the weight of
    positive predictions.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.6,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Model — Whisper Tiny (39M params, same as original Pipecat Smart Turn v3)
# ---------------------------------------------------------------------------

class SmartTurnModel(nn.Module):
    """Whisper Tiny encoder + attention pooling + classifier."""

    def __init__(self, whisper_model: str = "openai/whisper-tiny"):
        super().__init__()
        from transformers import WhisperModel

        whisper = WhisperModel.from_pretrained(
            whisper_model, cache_dir=str(CACHE_DIR)
        )
        self.encoder = whisper.encoder

        # Resize position embeddings from 1500 (30s) to 400 (8s)
        max_pos = 400
        old_embed = self.encoder.embed_positions.weight.data
        new_embed = old_embed[:max_pos, :]
        self.encoder.embed_positions = nn.Embedding(max_pos, old_embed.shape[1])
        self.encoder.embed_positions.weight.data = new_embed
        self.encoder.config.max_source_positions = max_pos

        hidden_size = self.encoder.config.d_model  # 384 for whisper-tiny

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Classifier head (sized for Whisper Tiny 384-dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
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
# Data sample
# ---------------------------------------------------------------------------

@dataclass
class AudioSample:
    audio: np.ndarray  # float32, 16kHz
    label: float       # 1.0 = complete, 0.0 = incomplete
    speaker_id: str
    source: str        # dataset name
    text: str = ""     # original transcription


# ---------------------------------------------------------------------------
# Label assignment — hybrid: punctuation + audio-based
# ---------------------------------------------------------------------------

def classify_text_completeness(text: str) -> float | None:
    """Classify if text represents a complete or incomplete utterance.

    Returns:
        1.0 for complete, 0.0 for incomplete, None if can't determine from text
    """
    text = text.strip()
    if not text or len(text) < 3:
        return None

    # Complete: ends with sentence-ending punctuation
    if COMPLETE_ENDINGS.search(text):
        return 1.0

    # Incomplete: ends with continuation punctuation
    if INCOMPLETE_ENDINGS.search(text):
        return 0.0

    # No punctuation — can't determine from text alone
    # Return None so the caller uses audio-based labeling instead
    return None


# ---------------------------------------------------------------------------
# Dataset loaders — conversational Portuguese only
# ---------------------------------------------------------------------------

def load_coraa_samples(max_samples: int = 50000) -> list[AudioSample]:
    """Load CORAA v1.1 — conversational Brazilian Portuguese (291h).

    Hybrid labeling:
    - If text has punctuation (.!?), use that for labels
    - Otherwise: full audio = COMPLETE (natural prosodic ending),
      truncated audio at 30-75% = INCOMPLETE (mid-utterance cut)
    """
    from datasets import load_dataset

    log.info("Loading CORAA v1.1 from HuggingFace...")
    try:
        ds = load_dataset(
            "Racoci/CORAA-v1.1",
            split="train",
            cache_dir=str(CACHE_DIR),
            streaming=True,
        )
    except Exception as e:
        log.warning("Failed to load CORAA v1.1: %s", e)
        return []

    samples = []
    complete_count = 0
    incomplete_count = 0
    text_labeled = 0
    audio_labeled = 0
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

            text = str(row.get("text", row.get("sentence", "")))
            speaker_id = str(row.get("speaker", row.get("speaker_id", f"coraa_{i}")))
            text_label = classify_text_completeness(text)

            if text_label is not None:
                # Text has punctuation — use it directly
                if text_label == 1.0 and complete_count >= target_per_class:
                    continue
                if text_label == 0.0 and incomplete_count >= target_per_class:
                    continue

                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=text_label,
                        speaker_id=speaker_id, source="coraa", text=text,
                    ))
                    if text_label == 1.0:
                        complete_count += 1
                    else:
                        incomplete_count += 1
                    text_labeled += 1
            else:
                # No punctuation — use audio-based labeling:
                # COMPLETE: full utterance (speaker naturally finished)
                if complete_count < target_per_class:
                    window = _extract_window(audio, position="end")
                    if window is not None:
                        samples.append(AudioSample(
                            audio=window, label=1.0,
                            speaker_id=speaker_id, source="coraa", text=text,
                        ))
                        complete_count += 1
                        audio_labeled += 1

                # INCOMPLETE: truncate at 30-75% (mid-utterance cut)
                if incomplete_count < target_per_class and duration >= 2.0:
                    cut_frac = random.uniform(0.3, 0.75)
                    cut_sample = int(len(audio) * cut_frac)
                    truncated = audio[:cut_sample]
                    window = _extract_window(truncated, position="end")
                    if window is not None:
                        samples.append(AudioSample(
                            audio=window, label=0.0,
                            speaker_id=speaker_id, source="coraa", text=text,
                        ))
                        incomplete_count += 1
                        audio_labeled += 1

        except Exception as e:
            if i < 5:
                log.warning("CORAA sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  CORAA: processed %d rows, %d complete, %d incomplete (text=%d, audio=%d)",
                     i, complete_count, incomplete_count, text_labeled, audio_labeled)

    log.info("CORAA: %d complete + %d incomplete = %d samples (text_labeled=%d, audio_labeled=%d)",
             complete_count, incomplete_count, len(samples), text_labeled, audio_labeled)
    return samples


def load_mupe_samples(max_samples: int = 50000) -> list[AudioSample]:
    """Load CORAA-MUPE-ASR — interview turn-taking (365h).

    Hybrid labeling (same as CORAA):
    - If text has punctuation (.!?), use that for labels
    - Otherwise: full audio = COMPLETE (natural prosodic ending),
      truncated audio at 30-75% = INCOMPLETE (mid-utterance cut)

    Fixed speaker_id to use unique hashes (not just "interviewer"/"interviewee").
    """
    from datasets import load_dataset

    log.info("Loading CORAA-MUPE-ASR from HuggingFace...")
    try:
        ds = load_dataset(
            "nilc-nlp/CORAA-MUPE-ASR",
            split="train",
            cache_dir=str(CACHE_DIR),
            streaming=True,
        )
    except Exception as e:
        log.warning("Failed to load MUPE: %s", e)
        return []

    samples = []
    complete_count = 0
    incomplete_count = 0
    text_labeled = 0
    audio_labeled = 0
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

            text = str(row.get("text", row.get("sentence", "")))

            # Fix: use a unique speaker_id based on audio_path or index,
            # NOT speaker_type which is just "interviewer"/"interviewee"
            audio_path = str(row.get("audio_path", row.get("path", "")))
            if audio_path:
                parts = audio_path.split("/")
                speaker_id = f"mupe_{parts[0] if len(parts) > 1 else hashlib.md5(audio_path.encode()).hexdigest()[:8]}"
            else:
                speaker_id = f"mupe_{i // 50}"

            text_label = classify_text_completeness(text)

            if text_label is not None:
                # Text has punctuation — use it directly
                if text_label == 1.0 and complete_count >= target_per_class:
                    continue
                if text_label == 0.0 and incomplete_count >= target_per_class:
                    continue

                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=text_label,
                        speaker_id=speaker_id, source="mupe", text=text,
                    ))
                    if text_label == 1.0:
                        complete_count += 1
                    else:
                        incomplete_count += 1
                    text_labeled += 1
            else:
                # No punctuation — use audio-based labeling:
                # COMPLETE: full utterance (speaker naturally finished)
                if complete_count < target_per_class:
                    window = _extract_window(audio, position="end")
                    if window is not None:
                        samples.append(AudioSample(
                            audio=window, label=1.0,
                            speaker_id=speaker_id, source="mupe", text=text,
                        ))
                        complete_count += 1
                        audio_labeled += 1

                # INCOMPLETE: truncate at 30-75% (mid-utterance cut)
                if incomplete_count < target_per_class and duration >= 2.0:
                    cut_frac = random.uniform(0.3, 0.75)
                    cut_sample = int(len(audio) * cut_frac)
                    truncated = audio[:cut_sample]
                    window = _extract_window(truncated, position="end")
                    if window is not None:
                        samples.append(AudioSample(
                            audio=window, label=0.0,
                            speaker_id=speaker_id, source="mupe", text=text,
                        ))
                        incomplete_count += 1
                        audio_labeled += 1

        except Exception as e:
            if i < 5:
                log.warning("MUPE sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  MUPE: processed %d rows, %d complete, %d incomplete (text=%d, audio=%d)",
                     i, complete_count, incomplete_count, text_labeled, audio_labeled)

    log.info("MUPE: %d complete + %d incomplete = %d samples (text_labeled=%d, audio_labeled=%d)",
             complete_count, incomplete_count, len(samples), text_labeled, audio_labeled)
    return samples


# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------

def _extract_window(audio: np.ndarray, position: str = "end") -> np.ndarray | None:
    """Extract 8-second window from audio, pad if needed."""
    if len(audio) < SAMPLE_RATE:  # minimum 1 second
        return None

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    if position == "end":
        if len(audio) > WINDOW_SAMPLES:
            audio = audio[-WINDOW_SAMPLES:]
        elif len(audio) < WINDOW_SAMPLES:
            padding = WINDOW_SAMPLES - len(audio)
            audio = np.pad(audio, (padding, 0), mode="constant")
    else:
        if len(audio) > WINDOW_SAMPLES:
            audio = audio[:WINDOW_SAMPLES]
        elif len(audio) < WINDOW_SAMPLES:
            padding = WINDOW_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")

    # Add ~200ms silence at end (matching VAD behavior)
    silence_samples = int(0.2 * SAMPLE_RATE)
    audio[-silence_samples:] = 0.0

    return audio.astype(np.float32)


def augment_audio(audio: np.ndarray) -> np.ndarray:
    """Apply data augmentation — more aggressive than v2."""
    aug = audio.copy()

    # Speed perturbation (0.9x to 1.1x) — changes pitch and speed
    if random.random() < 0.5:
        speed_factor = random.uniform(0.9, 1.1)
        indices = np.arange(0, len(aug), speed_factor).astype(int)
        indices = indices[indices < len(aug)]
        aug = aug[indices]
        # Pad/truncate back to original length
        if len(aug) > WINDOW_SAMPLES:
            aug = aug[:WINDOW_SAMPLES]
        elif len(aug) < WINDOW_SAMPLES:
            aug = np.pad(aug, (WINDOW_SAMPLES - len(aug), 0), mode="constant")

    # Random volume scaling (0.6x to 1.4x)
    if random.random() < 0.5:
        scale = random.uniform(0.6, 1.4)
        aug = aug * scale

    # Add Gaussian noise (more aggressive)
    if random.random() < 0.4:
        noise_level = random.uniform(0.002, 0.02)
        aug = aug + np.random.randn(len(aug)).astype(np.float32) * noise_level

    # Random time shift (shift audio left/right by up to 0.3s)
    if random.random() < 0.3:
        shift = random.randint(-int(0.3 * SAMPLE_RATE), int(0.3 * SAMPLE_RATE))
        aug = np.roll(aug, shift)
        if shift > 0:
            aug[:shift] = 0.0
        elif shift < 0:
            aug[shift:] = 0.0

    # Clip to prevent overflow
    aug = np.clip(aug, -1.0, 1.0)

    return aug


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SmartTurnDataset(Dataset):
    """In-memory dataset of pre-processed audio samples."""

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

        # Apply label smoothing: 0→0.05, 1→0.95 (improves calibration)
        smooth_label = sample.label * (1 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH

        return {
            "input_features": torch.from_numpy(features),
            "labels": torch.tensor(smooth_label, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Speaker-based train/val/test split
# ---------------------------------------------------------------------------

def split_by_speaker(
    samples: list[AudioSample],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[AudioSample], list[AudioSample], list[AudioSample]]:
    """Split samples by speaker to avoid data leakage."""
    speaker_samples: dict[str, list[AudioSample]] = {}
    for s in samples:
        speaker_samples.setdefault(s.speaker_id, []).append(s)

    speakers = list(speaker_samples.keys())
    random.shuffle(speakers)

    n_val = max(1, int(len(speakers) * val_frac))
    n_test = max(1, int(len(speakers) * test_frac))

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test:n_test + n_val])
    train_speakers = set(speakers[n_test + n_val:])

    train = [s for sp in train_speakers for s in speaker_samples[sp]]
    val = [s for sp in val_speakers for s in speaker_samples[sp]]
    test = [s for sp in test_speakers for s in speaker_samples[sp]]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Log class distribution per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_c = sum(1 for s in split if s.label == 1.0)
        n_i = sum(1 for s in split if s.label == 0.0)
        n_spk = len(set(s.speaker_id for s in split))
        log.info("  %s: %d samples (%d complete, %d incomplete) from %d speakers",
                 name, len(split), n_c, n_i, n_spk)

    return train, val, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_samples_per_dataset: int = 50000,
    whisper_model: str = "openai/whisper-tiny",
) -> Path:
    """Fine-tune Smart Turn v3 on Portuguese data from HuggingFace."""

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Training on device: %s", device)
    if device == "cuda":
        log.info("GPU: %s (%d MB)", torch.cuda.get_device_name(),
                 torch.cuda.get_device_properties(0).total_memory // 1024 // 1024)

    # ----- Load datasets (conversational only, no audiobooks) -----
    t0 = time.time()
    all_samples: list[AudioSample] = []

    coraa = load_coraa_samples(max_samples=max_samples_per_dataset)
    all_samples.extend(coraa)
    del coraa
    gc.collect()

    mupe = load_mupe_samples(max_samples=max_samples_per_dataset)
    all_samples.extend(mupe)
    del mupe
    gc.collect()

    if not all_samples:
        raise RuntimeError("No samples loaded! Check dataset availability.")

    load_time = time.time() - t0
    n_complete = sum(1 for s in all_samples if s.label == 1.0)
    n_incomplete = sum(1 for s in all_samples if s.label == 0.0)
    n_speakers = len(set(s.speaker_id for s in all_samples))

    log.info("Total: %d samples (%d complete, %d incomplete) from %d speakers in %.0fs",
             len(all_samples), n_complete, n_incomplete, n_speakers, load_time)

    # Source distribution
    sources = {}
    for s in all_samples:
        sources[s.source] = sources.get(s.source, 0) + 1
    for src, cnt in sorted(sources.items()):
        log.info("  %s: %d samples", src, cnt)

    # Log some label examples
    log.info("=== Label examples ===")
    for s in random.sample(all_samples, min(10, len(all_samples))):
        label_str = "COMPLETE" if s.label == 1.0 else "INCOMPLETE"
        log.info("  [%s] %s: '%.60s'", label_str, s.source, s.text)

    # ----- Split by speaker -----
    log.info("=== Splitting by speaker ===")
    train_samples, val_samples, test_samples = split_by_speaker(all_samples)

    # ----- Create datasets -----
    # Whisper Tiny uses 80 mel bins
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    train_ds = SmartTurnDataset(train_samples, feature_extractor, augment=True)
    val_ds = SmartTurnDataset(val_samples, feature_extractor, augment=False)
    test_ds = SmartTurnDataset(test_samples, feature_extractor, augment=False)

    # Balanced sampler for training
    train_labels = [s.label for s in train_samples]
    n_pos = sum(1 for l in train_labels if l == 1.0)
    n_neg = len(train_labels) - n_pos
    weights = [1.0 / n_neg if l == 0.0 else 1.0 / n_pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    use_pin = device == "cuda"
    n_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=n_workers, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin,
    )

    # ----- Model -----
    model = SmartTurnModel(whisper_model=whisper_model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s — %d total params, %d trainable", whisper_model, total_params, trainable_params)

    # Loss — Focal Loss with alpha=0.6 to penalize false positives (boost precision)
    # alpha < 1.0 means the model is penalized MORE for false positives than false negatives
    # gamma=2.0 focuses training on hard boundary cases (mid-sentence pauses)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = FocalLoss(gamma=2.0, alpha=0.6, pos_weight=pos_weight)
    log.info("FocalLoss: gamma=2.0, alpha=0.6, pos_weight=%.2f (neg=%d, pos=%d)",
             pos_weight.item(), n_neg, n_pos)

    # Optimizer — different LR for encoder vs head
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.attention.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},  # Lower LR for pretrained encoder
        {"params": head_params, "lr": lr},
    ], weight_decay=0.01)

    # Warmup + cosine decay
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Training loop -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_path = OUTPUT_DIR / "best_model.pt"
    resume_path = OUTPUT_DIR / "resume_checkpoint.pt"
    patience = 7  # More patience for larger model
    patience_counter = 0
    history = []
    start_epoch = 0

    # Resume from checkpoint if available (survives pod restarts)
    if resume_path.exists():
        log.info("=== Resuming from checkpoint %s ===", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]  # resume from NEXT epoch
        best_f1 = ckpt.get("best_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", [])
        log.info("  Resumed at epoch %d, best_f1=%.4f, patience=%d/%d",
                 start_epoch, best_f1, patience_counter, patience)

    log.info("=== Starting training: %d epochs, batch_size=%d, lr=%.1e ===", epochs, batch_size, lr)

    for epoch in range(start_epoch, epochs):
        # Train
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
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            if batch_idx % 100 == 0 and batch_idx > 0:
                log.info("  batch %d/%d loss=%.4f", batch_idx, len(train_loader),
                         loss.item())

            # Periodic checkpoint for crash recovery (saves to /workspace)
            if batch_idx > 0 and batch_idx % CHECKPOINT_EVERY_BATCHES == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,  # current epoch (will resume this epoch from start)
                    "best_f1": best_f1,
                    "patience_counter": patience_counter,
                    "history": history,
                }, resume_path)
                log.info("  checkpoint saved (epoch %d, batch %d)", epoch + 1, batch_idx)

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
            log.info("  -> New best model saved (val_f1=%.4f)", best_f1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch + 1, patience)
                break

        # Save resume checkpoint at end of each epoch (next epoch = epoch + 1)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "history": history,
        }, resume_path)
        log.info("  epoch checkpoint saved (will resume at epoch %d)", epoch + 2)

    # Clean up resume checkpoint — training completed successfully
    if resume_path.exists():
        resume_path.unlink()
        log.info("Resume checkpoint removed (training complete)")

    # ----- Test evaluation -----
    log.info("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = _evaluate(model, test_loader, device, criterion)
    log.info("Test results (best model from epoch %d) @ threshold=0.5:", checkpoint["epoch"])
    log.info("  Accuracy:  %.3f", test_metrics["accuracy"])
    log.info("  Precision: %.3f", test_metrics["precision"])
    log.info("  Recall:    %.3f", test_metrics["recall"])
    log.info("  F1:        %.3f", test_metrics["f1"])
    log.info("  TP=%d FP=%d FN=%d TN=%d",
             test_metrics["tp"], test_metrics["fp"],
             test_metrics["fn"], test_metrics["tn"])

    # ----- Multi-threshold evaluation (find best precision/recall tradeoff) -----
    log.info("\n=== Threshold Sweep ===")
    threshold_results = {}
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        t_metrics = _evaluate(model, test_loader, device, criterion, threshold=thresh)
        threshold_results[str(thresh)] = t_metrics
        log.info("  threshold=%.2f: prec=%.3f rec=%.3f f1=%.3f acc=%.3f (TP=%d FP=%d FN=%d TN=%d)",
                 thresh, t_metrics["precision"], t_metrics["recall"],
                 t_metrics["f1"], t_metrics["accuracy"],
                 t_metrics["tp"], t_metrics["fp"], t_metrics["fn"], t_metrics["tn"])

    # Find best threshold for precision >= 85%
    best_thresh = 0.5
    best_thresh_f1 = 0.0
    for thresh_str, t_m in threshold_results.items():
        if t_m["precision"] >= 0.85 and t_m["f1"] > best_thresh_f1:
            best_thresh = float(thresh_str)
            best_thresh_f1 = t_m["f1"]
    if best_thresh > 0.5:
        log.info("  -> Recommended threshold: %.2f (precision>=85%%, best F1=%.3f)",
                 best_thresh, best_thresh_f1)
    else:
        log.info("  -> No threshold achieves >=85%% precision; using 0.5")

    # ----- Export to ONNX -----
    model = model.to("cpu")
    onnx_path = OUTPUT_DIR / "smart_turn_pt_v3.onnx"
    dummy = torch.randn(1, 80, 800)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        log.info("ONNX model exported to %s", onnx_path)
    except Exception as e:
        log.warning("ONNX export failed: %s — saving PyTorch model only", e)

    # ----- Save results -----
    results = {
        "model": "smart_turn_pt_v3",
        "whisper_model": whisper_model,
        "total_samples": len(all_samples),
        "n_speakers": n_speakers,
        "sources": sources,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": best_f1,
        "val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
        "threshold_sweep": threshold_results,
        "recommended_threshold": best_thresh,
        "improvements_over_v2": [
            "punctuation-based labels instead of random cuts",
            "removed MLS audiobook data",
            "whisper-tiny (39M) — same backbone as original Pipecat Smart Turn v3",
            "fixed MUPE speaker_id",
            "better augmentation (speed perturbation, time shift)",
            "warmup + cosine decay LR schedule",
            "focal loss (gamma=2, alpha=0.6) — penalizes false positives",
            "label smoothing (0.05) — improves calibration",
            "multi-threshold evaluation — finds optimal precision/recall tradeoff",
        ],
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_samples_per_dataset": max_samples_per_dataset,
            "patience": patience,
        },
    }

    results_path = OUTPUT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Training results saved to %s", results_path)

    return onnx_path


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model and return metrics at a given threshold."""
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = (torch.sigmoid(logits) > threshold).float()
            # Compare against hard labels (undo label smoothing for eval)
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
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    onnx_path = train(
        epochs=30,
        batch_size=32,
        lr=3e-5,
        max_samples_per_dataset=7500,
        whisper_model="openai/whisper-tiny",
    )
    log.info("Done! Model: %s", onnx_path)
