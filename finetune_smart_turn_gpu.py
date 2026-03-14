"""
Fine-tune Pipecat Smart Turn on Portuguese speech data (GPU version).

Downloads Portuguese datasets from HuggingFace, processes into
complete/incomplete samples, trains with speaker-based splits
to avoid data leakage, and exports to ONNX.

Datasets used:
- CORAA v1.1 (291h, conversational Brazilian Portuguese)
- Common Voice Portuguese (51h, diverse speakers)
- MLS Portuguese (168h, read speech)

Run on a Vast.ai GPU instance with:
    python finetune_smart_turn_gpu.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
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

OUTPUT_DIR = Path("checkpoints/smart_turn_pt_v2")
CACHE_DIR = Path("hf_cache")


# ---------------------------------------------------------------------------
# Model (same architecture as Pipecat Smart Turn v3)
# ---------------------------------------------------------------------------

class SmartTurnModel(nn.Module):
    """Whisper Tiny encoder + attention pooling + classifier."""

    def __init__(self):
        super().__init__()
        from transformers import WhisperModel

        whisper = WhisperModel.from_pretrained(
            "openai/whisper-tiny", cache_dir=str(CACHE_DIR)
        )
        self.encoder = whisper.encoder

        # Resize position embeddings from 1500 (30s) to 400 (8s)
        max_pos = 400
        old_embed = self.encoder.embed_positions.weight.data
        new_embed = old_embed[:max_pos, :]
        self.encoder.embed_positions = nn.Embedding(max_pos, old_embed.shape[1])
        self.encoder.embed_positions.weight.data = new_embed
        self.encoder.config.max_source_positions = max_pos

        hidden_size = self.encoder.config.d_model  # 384

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

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
# Data sample
# ---------------------------------------------------------------------------

@dataclass
class AudioSample:
    audio: np.ndarray  # float32, 16kHz
    label: float       # 1.0 = complete, 0.0 = incomplete
    speaker_id: str
    source: str        # dataset name


# ---------------------------------------------------------------------------
# Dataset loaders — all Portuguese
# ---------------------------------------------------------------------------

def load_coraa_samples(max_samples: int = 30000) -> list[AudioSample]:
    """Load CORAA v1.1 — conversational Brazilian Portuguese (291h)."""
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

            # Resample to 16kHz if needed
            if sr != SAMPLE_RATE:
                import torchaudio
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE).squeeze().numpy()

            duration = len(audio) / SAMPLE_RATE
            if duration < 1.0:
                continue

            speaker_id = str(row.get("speaker", row.get("speaker_id", f"coraa_{i}")))

            # COMPLETE: use the end of the utterance (last 8s)
            if complete_count < target_per_class:
                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        speaker_id=speaker_id, source="coraa",
                    ))
                    complete_count += 1

            # INCOMPLETE: use a random mid-utterance cut (first 40-80%)
            if incomplete_count < target_per_class and duration >= 2.0:
                cut_frac = random.uniform(0.3, 0.75)
                cut_sample = int(len(audio) * cut_frac)
                truncated = audio[:cut_sample]
                window = _extract_window(truncated, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        speaker_id=speaker_id, source="coraa",
                    ))
                    incomplete_count += 1

        except Exception as e:
            if i < 5:
                log.warning("CORAA sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  CORAA: processed %d rows, %d complete, %d incomplete",
                     i, complete_count, incomplete_count)

    log.info("CORAA: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
    return samples


def load_common_voice_samples(max_samples: int = 20000) -> list[AudioSample]:
    """Load Common Voice Portuguese — diverse speakers (51h)."""
    from datasets import load_dataset

    log.info("Loading Common Voice Portuguese from HuggingFace...")
    try:
        ds = load_dataset(
            "fsicoli/common_voice_22_0", "pt",
            split="train",
            cache_dir=str(CACHE_DIR),
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback to mozilla-foundation version
        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0", "pt",
                split="train",
                cache_dir=str(CACHE_DIR),
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            log.warning("Failed to load Common Voice: %s", e)
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

            speaker_id = str(row.get("client_id", f"cv_{i}"))

            # COMPLETE: each CV sample is a full sentence
            if complete_count < target_per_class:
                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        speaker_id=speaker_id, source="common_voice",
                    ))
                    complete_count += 1

            # INCOMPLETE: cut at 30-70% of the sentence
            if incomplete_count < target_per_class and duration >= 1.5:
                cut_frac = random.uniform(0.3, 0.7)
                cut_sample = int(len(audio) * cut_frac)
                truncated = audio[:cut_sample]
                window = _extract_window(truncated, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        speaker_id=speaker_id, source="common_voice",
                    ))
                    incomplete_count += 1

        except Exception as e:
            if i < 5:
                log.warning("CV sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  CV: processed %d rows, %d complete, %d incomplete",
                     i, complete_count, incomplete_count)

    log.info("Common Voice: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
    return samples


def load_mls_samples(max_samples: int = 20000) -> list[AudioSample]:
    """Load MLS Portuguese — read speech from audiobooks (168h)."""
    from datasets import load_dataset

    log.info("Loading MLS Portuguese from HuggingFace...")
    try:
        ds = load_dataset(
            "facebook/multilingual_librispeech", "portuguese",
            split="train",
            cache_dir=str(CACHE_DIR),
            streaming=True,
        )
    except Exception as e:
        log.warning("Failed to load MLS: %s", e)
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

            speaker_id = str(row.get("speaker_id", f"mls_{i}"))

            # COMPLETE
            if complete_count < target_per_class:
                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        speaker_id=speaker_id, source="mls",
                    ))
                    complete_count += 1

            # INCOMPLETE
            if incomplete_count < target_per_class and duration >= 1.5:
                cut_frac = random.uniform(0.3, 0.7)
                cut_sample = int(len(audio) * cut_frac)
                truncated = audio[:cut_sample]
                window = _extract_window(truncated, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        speaker_id=speaker_id, source="mls",
                    ))
                    incomplete_count += 1

        except Exception as e:
            if i < 5:
                log.warning("MLS sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  MLS: processed %d rows, %d complete, %d incomplete",
                     i, complete_count, incomplete_count)

    log.info("MLS: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
    return samples


def load_mupe_samples(max_samples: int = 20000) -> list[AudioSample]:
    """Load CORAA-MUPE-ASR — interview turn-taking (365h)."""
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

            speaker_id = str(row.get("speaker_type", f"mupe_{i}"))

            # COMPLETE
            if complete_count < target_per_class:
                window = _extract_window(audio, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=1.0,
                        speaker_id=speaker_id, source="mupe",
                    ))
                    complete_count += 1

            # INCOMPLETE
            if incomplete_count < target_per_class and duration >= 2.0:
                cut_frac = random.uniform(0.3, 0.75)
                cut_sample = int(len(audio) * cut_frac)
                truncated = audio[:cut_sample]
                window = _extract_window(truncated, position="end")
                if window is not None:
                    samples.append(AudioSample(
                        audio=window, label=0.0,
                        speaker_id=speaker_id, source="mupe",
                    ))
                    incomplete_count += 1

        except Exception as e:
            if i < 5:
                log.warning("MUPE sample %d error: %s", i, e)
            continue

        if i % 5000 == 0 and i > 0:
            log.info("  MUPE: processed %d rows, %d complete, %d incomplete",
                     i, complete_count, incomplete_count)

    log.info("MUPE: %d complete + %d incomplete = %d samples",
             complete_count, incomplete_count, len(samples))
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
        # Take last 8 seconds
        if len(audio) > WINDOW_SAMPLES:
            audio = audio[-WINDOW_SAMPLES:]
        elif len(audio) < WINDOW_SAMPLES:
            padding = WINDOW_SAMPLES - len(audio)
            audio = np.pad(audio, (padding, 0), mode="constant")
    else:
        # Take first 8 seconds
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
    """Apply simple data augmentation."""
    aug = audio.copy()

    # Random volume scaling (0.7x to 1.3x)
    if random.random() < 0.5:
        scale = random.uniform(0.7, 1.3)
        aug = aug * scale

    # Add small Gaussian noise
    if random.random() < 0.3:
        noise_level = random.uniform(0.001, 0.01)
        aug = aug + np.random.randn(len(aug)).astype(np.float32) * noise_level

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

        return {
            "input_features": torch.from_numpy(features),
            "labels": torch.tensor(sample.label, dtype=torch.float32),
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
    # Group by speaker
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

    log.info("Split: %d train (%d speakers), %d val (%d speakers), %d test (%d speakers)",
             len(train), len(train_speakers), len(val), len(val_speakers),
             len(test), len(test_speakers))

    return train, val, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_samples_per_dataset: int = 25000,
) -> Path:
    """Fine-tune Smart Turn on Portuguese data from HuggingFace."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training on device: %s", device)
    if device == "cuda":
        log.info("GPU: %s (%d MB)", torch.cuda.get_device_name(),
                 torch.cuda.get_device_properties(0).total_mem // 1024 // 1024)

    # ----- Load datasets -----
    t0 = time.time()
    all_samples: list[AudioSample] = []

    # Load from multiple Portuguese sources
    coraa = load_coraa_samples(max_samples=max_samples_per_dataset)
    all_samples.extend(coraa)
    del coraa
    gc.collect()

    cv = load_common_voice_samples(max_samples=max_samples_per_dataset)
    all_samples.extend(cv)
    del cv
    gc.collect()

    mls = load_mls_samples(max_samples=max_samples_per_dataset)
    all_samples.extend(mls)
    del mls
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

    # ----- Split by speaker -----
    train_samples, val_samples, test_samples = split_by_speaker(all_samples)

    # ----- Create datasets -----
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ----- Model -----
    model = SmartTurnModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d total params, %d trainable", total_params, trainable_params)

    # Loss
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info("pos_weight: %.2f", pos_weight.item())

    # Optimizer — different LR for encoder vs head
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.attention.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},  # Lower LR for pretrained encoder
        {"params": head_params, "lr": lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ----- Training loop -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_path = OUTPUT_DIR / "best_model.pt"
    patience = 5
    patience_counter = 0
    history = []

    for epoch in range(epochs):
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

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            if batch_idx % 100 == 0 and batch_idx > 0:
                log.info("  batch %d/%d loss=%.4f", batch_idx, len(train_loader),
                         loss.item())

        scheduler.step()

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
            }, best_path)
            log.info("  -> New best model saved (val_f1=%.3f)", best_f1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch + 1, patience)
                break

    # ----- Test evaluation -----
    log.info("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = _evaluate(model, test_loader, device, criterion)
    log.info("Test results:")
    log.info("  Accuracy:  %.3f", test_metrics["accuracy"])
    log.info("  Precision: %.3f", test_metrics["precision"])
    log.info("  Recall:    %.3f", test_metrics["recall"])
    log.info("  F1:        %.3f", test_metrics["f1"])
    log.info("  TP=%d FP=%d FN=%d TN=%d",
             test_metrics["tp"], test_metrics["fp"],
             test_metrics["fn"], test_metrics["tn"])

    # ----- Export to ONNX -----
    model = model.to("cpu")
    onnx_path = OUTPUT_DIR / "smart_turn_pt_v2.onnx"
    dummy = torch.randn(1, 80, 800)
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

    # ----- Save results -----
    results = {
        "model": "smart_turn_pt_v2",
        "total_samples": len(all_samples),
        "n_speakers": n_speakers,
        "sources": sources,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_samples_per_dataset": max_samples_per_dataset,
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
) -> dict:
    """Evaluate model and return metrics."""
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

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

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
        epochs=20,
        batch_size=32,
        lr=2e-5,
        max_samples_per_dataset=25000,
    )
    log.info("Done! ONNX model: %s", onnx_path)
