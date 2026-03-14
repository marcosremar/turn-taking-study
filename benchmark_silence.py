"""
Baseline: Silence-threshold turn-taking detection.

The simplest approach — detect turn boundaries by measuring silence duration.
This serves as the lower-bound baseline for comparison.

Reference:
- Raux, A. & Eskenazi, M. (2009). A Finite-State Turn-Taking Model for
  Spoken Dialog Systems. NAACL-HLT 2009.
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from benchmark_base import TurnTakingModel, PredictedEvent
from setup_dataset import Conversation

log = logging.getLogger(__name__)


class SilenceThresholdModel(TurnTakingModel):
    """Detect turn shifts based on silence duration exceeding a threshold."""

    def __init__(self, silence_threshold_ms: float = 700.0, energy_threshold: float = 0.01):
        self.silence_threshold_ms = silence_threshold_ms
        self.energy_threshold = energy_threshold

    @property
    def name(self) -> str:
        return f"silence_{int(self.silence_threshold_ms)}ms"

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def requires_asr(self) -> bool:
        return False

    def get_model_size_mb(self) -> float:
        return 0.0  # No model

    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        events: list[PredictedEvent] = []

        if conversation.audio_path:
            return self._predict_from_audio(conversation)

        # Fallback: predict from turn annotations (text-only dataset)
        return self._predict_from_turns(conversation)

    def _predict_from_audio(self, conversation: Conversation) -> list[PredictedEvent]:
        """Detect silence periods in audio and predict turn shifts."""
        audio, sr = sf.read(conversation.audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        events: list[PredictedEvent] = []
        frame_size = int(0.032 * sr)  # 32ms frames
        threshold_frames = int(self.silence_threshold_ms / 32.0)

        silent_frames = 0
        was_active = False
        last_active_end = 0.0

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms < self.energy_threshold:
                silent_frames += 1
                if was_active and silent_frames >= threshold_frames:
                    # Silence exceeded threshold — predict turn shift
                    shift_time = last_active_end + self.silence_threshold_ms / 1000.0
                    events.append(PredictedEvent(
                        timestamp=shift_time,
                        event_type="shift",
                        confidence=min(1.0, silent_frames / threshold_frames),
                    ))
                    was_active = False
            else:
                if silent_frames > 0 and silent_frames < threshold_frames:
                    # Short pause — hold
                    events.append(PredictedEvent(
                        timestamp=i / sr,
                        event_type="hold",
                        confidence=1.0 - (silent_frames / threshold_frames),
                    ))
                silent_frames = 0
                was_active = True
                last_active_end = (i + frame_size) / sr

        return events

    def _predict_from_turns(self, conversation: Conversation) -> list[PredictedEvent]:
        """Predict from turn timing annotations (when no audio available)."""
        events: list[PredictedEvent] = []
        threshold_s = self.silence_threshold_ms / 1000.0

        for i in range(1, len(conversation.turns)):
            gap = conversation.turns[i].start - conversation.turns[i - 1].end
            if gap >= threshold_s:
                events.append(PredictedEvent(
                    timestamp=conversation.turns[i].start,
                    event_type="shift",
                    confidence=min(1.0, gap / (threshold_s * 2)),
                ))
            else:
                events.append(PredictedEvent(
                    timestamp=conversation.turns[i].start,
                    event_type="hold",
                    confidence=max(0.0, 1.0 - gap / threshold_s),
                ))

        return events
