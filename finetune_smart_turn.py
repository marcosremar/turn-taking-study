"""
Fine-tune Pipecat Smart Turn on Portuguese data.

Loads the pretrained Whisper Tiny encoder + classifier, then continues
training on Portuguese audio samples from NURC-SP and Edge TTS.

Can run on MPS (Apple Silicon), CUDA, or CPU.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

import soundfile as sf
import onnxruntime as ort

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 8 * SAMPLE_RATE

DATA_DIR = Path(__file__).parent / "data" / "smart_turn_pt_training" / "por"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "smart_turn_pt"


class SmartTurnModel(nn.Module):
    """Whisper encoder + attention pooling + classifier (matches Smart Turn v3 architecture)."""

    def __init__(self):
        super().__init__()
        from transformers import WhisperModel, WhisperConfig

        whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
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

        # Classifier
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
        # input_features: (batch, 80, time_frames)
        encoder_output = self.encoder(input_features).last_hidden_state  # (batch, seq, 384)

        # Attention pooling
        attn_weights = self.attention(encoder_output)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (encoder_output * attn_weights).sum(dim=1)  # (batch, 384)

        # Classify
        logits = self.classifier(pooled)  # (batch, 1)
        return logits.squeeze(-1)


class PortugueseDataset(Dataset):
    """Load Portuguese training samples from FLAC files."""

    def __init__(self, data_dir: Path, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.samples = []

        # Load complete samples (label=1)
        complete_dir = data_dir / "complete-nofiller"
        if complete_dir.exists():
            for f in sorted(complete_dir.glob("*.flac")):
                self.samples.append((str(f), 1.0))

        # Load incomplete samples (label=0)
        incomplete_dir = data_dir / "incomplete-nofiller"
        if incomplete_dir.exists():
            for f in sorted(incomplete_dir.glob("*.flac")):
                self.samples.append((str(f), 0.0))

        log.info("Loaded %d samples (%d complete, %d incomplete)",
                 len(self.samples),
                 sum(1 for _, l in self.samples if l == 1.0),
                 sum(1 for _, l in self.samples if l == 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Truncate/pad to 8 seconds
        if len(audio) > WINDOW_SAMPLES:
            audio = audio[-WINDOW_SAMPLES:]
        elif len(audio) < WINDOW_SAMPLES:
            padding = WINDOW_SAMPLES - len(audio)
            audio = np.pad(audio, (padding, 0), mode="constant")

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
            "labels": torch.tensor(label, dtype=torch.float32),
        }


def train(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "auto",
) -> Path:
    """Fine-tune Smart Turn on Portuguese data."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Training on device: %s", device)

    # Model
    model = SmartTurnModel()
    model = model.to(device)

    # Dataset
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    dataset = PortugueseDataset(DATA_DIR, feature_extractor)

    # Split 90/10
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info("Train: %d samples, Val: %d samples", n_train, n_val)

    # Loss with dynamic pos_weight
    n_pos = sum(1 for _, l in dataset.samples if l == 1.0)
    n_neg = len(dataset.samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info("pos_weight: %.2f (neg=%d, pos=%d)", pos_weight.item(), n_neg, n_pos)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_fn = val_tn = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                logits = model(features)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        log.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.3f val_acc=%.3f "
            "prec=%.3f rec=%.3f f1=%.3f",
            epoch + 1, epochs,
            train_loss / max(train_total, 1),
            train_acc, val_acc, precision, recall, f1,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            log.info("  -> New best model saved (val_acc=%.3f)", best_acc)

    log.info("Training complete. Best val_acc=%.3f", best_acc)

    # Export to ONNX
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    model = model.to("cpu")

    onnx_path = OUTPUT_DIR / "smart_turn_pt.onnx"
    dummy = torch.randn(1, 80, 800)  # (batch, mel_bins, frames) for 8s
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

    return onnx_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    onnx_path = train(epochs=15, batch_size=16, lr=2e-5)
    log.info("Done! ONNX model: %s", onnx_path)
