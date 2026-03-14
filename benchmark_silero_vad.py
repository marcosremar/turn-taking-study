"""
Silero VAD-based turn-taking detection.

Uses Silero VAD (Voice Activity Detection) to detect speech segments,
then infers turn-taking events from gaps between speech segments.
This represents BabelCast's current approach.

Reference:
- Silero Team. (2021). Silero VAD: pre-trained enterprise-grade Voice
  Activity Detector. https://github.com/snakers4/silero-vad
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf
import torch

from benchmark_base import TurnTakingModel, PredictedEvent
from setup_dataset import Conversation

log = logging.getLogger(__name__)


class SileroVADModel(TurnTakingModel):
    """Turn-taking detection using Silero VAD speech segments."""

    def __init__(
        self,
        threshold: float = 0.35,
        min_silence_ms: float = 300.0,
        min_speech_ms: float = 400.0,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self._model = None
        self._utils = None

    @property
    def name(self) -> str:
        return "silero_vad"

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def requires_asr(self) -> bool:
        return False

    def get_model_size_mb(self) -> float:
        return 2.0  # ~2MB ONNX model

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False
        )

    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        if not conversation.audio_path:
            return self._predict_from_turns(conversation)

        self._load_model()
        return self._predict_from_audio(conversation)

    def _predict_from_audio(self, conversation: Conversation) -> list[PredictedEvent]:
        """Run Silero VAD on audio and detect turn boundaries."""
        audio, sr = sf.read(conversation.audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze().numpy()
            sr = 16000

        get_speech_timestamps = self._utils[0]
        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            min_silence_duration_ms=int(self.min_silence_ms),
            min_speech_duration_ms=int(self.min_speech_ms),
            sampling_rate=sr,
        )

        events: list[PredictedEvent] = []
        min_silence_s = self.min_silence_ms / 1000.0

        for i in range(1, len(speech_timestamps)):
            prev_end = speech_timestamps[i - 1]["end"] / sr
            curr_start = speech_timestamps[i]["start"] / sr
            gap = curr_start - prev_end

            if gap >= min_silence_s:
                events.append(PredictedEvent(
                    timestamp=curr_start,
                    event_type="shift",
                    confidence=min(1.0, gap / (min_silence_s * 3)),
                ))
            else:
                events.append(PredictedEvent(
                    timestamp=curr_start,
                    event_type="hold",
                    confidence=max(0.0, 1.0 - gap / min_silence_s),
                ))

        return events

    def _predict_from_turns(self, conversation: Conversation) -> list[PredictedEvent]:
        """Fallback: simulate VAD behavior from turn annotations."""
        events: list[PredictedEvent] = []
        min_silence_s = self.min_silence_ms / 1000.0

        for i in range(1, len(conversation.turns)):
            gap = conversation.turns[i].start - conversation.turns[i - 1].end
            if gap >= min_silence_s:
                events.append(PredictedEvent(
                    timestamp=conversation.turns[i].start,
                    event_type="shift",
                    confidence=min(1.0, gap / (min_silence_s * 3)),
                ))
            else:
                events.append(PredictedEvent(
                    timestamp=conversation.turns[i].start,
                    event_type="hold",
                    confidence=max(0.0, 1.0 - gap / min_silence_s),
                ))

        return events
