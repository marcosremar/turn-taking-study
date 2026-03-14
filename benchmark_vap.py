"""
Voice Activity Projection (VAP) turn-taking benchmark.

VAP is a self-supervised model that predicts future voice activity for both
speakers in a dyadic dialogue, using only audio input.

References:
- Ekstedt, E. & Torre, G. (2024). Real-time and Continuous Turn-taking
  Prediction Using Voice Activity Projection. arXiv:2401.04868.
- Ekstedt, E. & Torre, G. (2022). Voice Activity Projection: Self-supervised
  Learning of Turn-taking Events. INTERSPEECH 2022.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from benchmark_base import TurnTakingModel, PredictedEvent
from setup_dataset import Conversation

log = logging.getLogger(__name__)

VAP_REPO = "/workspace/vap"
VAP_CHECKPOINT_URL = "https://huggingface.co/erikekstedt/vap/resolve/main/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"


class VAPModel(TurnTakingModel):
    """Voice Activity Projection model for turn-taking prediction."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None

    @property
    def name(self) -> str:
        return "vap"

    @property
    def requires_gpu(self) -> bool:
        return False  # Can run on CPU, but faster on GPU

    @property
    def requires_asr(self) -> bool:
        return False  # Audio-only

    def get_model_size_mb(self) -> float:
        return 20.0  # ~20MB

    def _ensure_installed(self) -> bool:
        """Check if VAP is installed, try to install if not."""
        try:
            import vap  # noqa: F401
            return True
        except ImportError:
            if Path(VAP_REPO).exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", VAP_REPO],
                    check=True, capture_output=True,
                )
                return True
            log.error(
                "VAP not installed. Clone https://github.com/ErikEkstedt/VoiceActivityProjection "
                "to %s and run: pip install -e %s", VAP_REPO, VAP_REPO
            )
            return False

    def _download_checkpoint(self) -> str:
        """Download pretrained VAP checkpoint."""
        ckpt_dir = Path(__file__).parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "vap_pretrained.pt"

        if not ckpt_path.exists():
            log.info("Downloading VAP checkpoint...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="erikekstedt/vap",
                filename="VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
                local_dir=str(ckpt_dir),
            )
            if Path(downloaded) != ckpt_path:
                os.rename(downloaded, ckpt_path)
            log.info("Checkpoint saved to %s", ckpt_path)

        return str(ckpt_path)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self._ensure_installed():
            raise RuntimeError("VAP model not available")

        from vap.model import VAPModel as VAPModelClass

        ckpt = self.checkpoint_path or self._download_checkpoint()
        log.info("Loading VAP model from %s on %s", ckpt, self.device)

        self._model = VAPModelClass.load_from_checkpoint(ckpt, map_location=self.device)
        self._model.eval()
        self._model.to(self.device)

    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        if not conversation.audio_path:
            log.warning("VAP requires audio — skipping %s", conversation.conv_id)
            return []

        self._load_model()

        audio, sr = sf.read(conversation.audio_path)

        # VAP expects stereo (2 channels, one per speaker)
        if audio.ndim == 1:
            # Mono: duplicate to stereo (model will still predict)
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2:
            if audio.shape[1] == 2:
                audio = audio.T  # (samples, 2) -> (2, samples)
            elif audio.shape[0] != 2:
                audio = np.stack([audio[0], audio[0]], axis=0)

        # Resample to 16kHz
        if sr != 16000:
            import torchaudio
            tensor = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(sr, 16000)
            tensor = resampler(tensor)
            audio = tensor.numpy()
            sr = 16000

        # Process in chunks (VAP uses 20s windows)
        chunk_samples = 20 * sr
        events: list[PredictedEvent] = []

        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, 2, samples)
        waveform = waveform.to(self.device)

        n_chunks = max(1, waveform.shape[-1] // chunk_samples)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_samples
            end = min(start + chunk_samples, waveform.shape[-1])
            chunk = waveform[:, :, start:end]

            if chunk.shape[-1] < sr:  # Skip very short chunks
                continue

            with torch.no_grad():
                output = self._model(chunk)

            # Extract turn-shift probabilities from VAP output
            # VAP outputs p_now and p_future for each speaker
            if hasattr(output, "p_now"):
                p = output.p_now.cpu().numpy().squeeze()
            elif isinstance(output, dict) and "p_now" in output:
                p = output["p_now"].cpu().numpy().squeeze()
            elif isinstance(output, tuple):
                p = output[0].cpu().numpy().squeeze()
            else:
                p = output.cpu().numpy().squeeze()

            chunk_events = self._extract_events(p, start / sr, sr)
            events.extend(chunk_events)

        return events

    def _extract_events(
        self,
        probs: np.ndarray,
        time_offset: float,
        sr: int,
        frame_hz: int = 50,
    ) -> list[PredictedEvent]:
        """Extract turn-taking events from VAP probability output."""
        events: list[PredictedEvent] = []

        if probs.ndim < 2:
            return events

        # probs shape: (n_frames, n_classes) or (n_frames, 2)
        n_frames = probs.shape[0]

        # Detect speaker dominance changes
        if probs.shape[-1] >= 2:
            speaker_a = probs[:, 0]
            speaker_b = probs[:, 1]
        else:
            return events

        prev_dominant = 0 if speaker_a[0] > speaker_b[0] else 1
        min_gap_frames = int(0.2 * frame_hz)  # 200ms minimum gap
        frames_since_change = min_gap_frames

        for i in range(1, n_frames):
            curr_dominant = 0 if speaker_a[i] > speaker_b[i] else 1
            frames_since_change += 1

            if curr_dominant != prev_dominant and frames_since_change >= min_gap_frames:
                timestamp = time_offset + i / frame_hz
                confidence = float(abs(speaker_a[i] - speaker_b[i]))

                events.append(PredictedEvent(
                    timestamp=timestamp,
                    event_type="shift",
                    confidence=confidence,
                ))
                prev_dominant = curr_dominant
                frames_since_change = 0
            elif curr_dominant == prev_dominant and frames_since_change == min_gap_frames:
                timestamp = time_offset + i / frame_hz
                events.append(PredictedEvent(
                    timestamp=timestamp,
                    event_type="hold",
                    confidence=float(max(speaker_a[i], speaker_b[i])),
                ))

        return events
