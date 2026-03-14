"""
Pipecat Smart Turn v3.1 turn-taking benchmark.

Smart Turn is a Whisper Tiny encoder + linear classifier that predicts
whether a speech segment is "complete" (turn ended) or "incomplete"
(speaker still talking). It processes 8-second audio windows.

References:
- Pipecat AI. (2025). Smart Turn: Real-time End-of-Turn Detection.
  https://github.com/pipecat-ai/smart-turn
- Model: pipecat-ai/smart-turn-v3 on HuggingFace.
  Trained on 23 languages including Portuguese.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from benchmark_base import TurnTakingModel, PredictedEvent
from setup_dataset import Conversation

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE


def _truncate_or_pad(audio: np.ndarray) -> np.ndarray:
    """Truncate to last 8 seconds or pad with zeros at start."""
    if len(audio) > WINDOW_SAMPLES:
        return audio[-WINDOW_SAMPLES:]
    elif len(audio) < WINDOW_SAMPLES:
        padding = WINDOW_SAMPLES - len(audio)
        return np.pad(audio, (padding, 0), mode="constant", constant_values=0)
    return audio


class PipecatSmartTurnModel(TurnTakingModel):
    """Pipecat Smart Turn v3.1 end-of-turn detector."""

    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        self.threshold = threshold
        self._model_path = model_path
        self._session: ort.InferenceSession | None = None
        self._feature_extractor = None

    @property
    def name(self) -> str:
        return "pipecat_smart_turn_v3.1"

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def requires_asr(self) -> bool:
        return False  # Audio-only (Whisper encoder, no decoder)

    def get_model_size_mb(self) -> float:
        return 8.0  # int8 ONNX

    def _load_model(self) -> None:
        if self._session is not None:
            return

        from transformers import WhisperFeatureExtractor

        # Download model if not provided
        if self._model_path is None:
            from huggingface_hub import hf_hub_download
            self._model_path = hf_hub_download(
                "pipecat-ai/smart-turn-v3", "smart-turn-v3.1-cpu.onnx"
            )

        log.info("Loading Pipecat Smart Turn from %s", self._model_path)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(self._model_path, sess_options=so)

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    def _predict_audio(self, audio: np.ndarray) -> dict:
        """Run Smart Turn inference on an audio array (16kHz mono)."""
        audio = _truncate_or_pad(audio)

        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )

        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        outputs = self._session.run(None, {"input_features": input_features})
        probability = outputs[0][0].item()

        return {
            "prediction": 1 if probability > self.threshold else 0,
            "probability": probability,
        }

    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        """Predict turn-taking events using Smart Turn.

        Strategy:
        1. At each turn boundary: extract 8s window ending there → model says
           "Complete" (shift) or "Incomplete" (hold).
        2. At mid-turn points (50% through each turn): these are ground truth
           "holds" (speaker is still talking). The model should predict
           "Incomplete" here. This gives the evaluation both classes.
        """
        if not conversation.audio_path:
            return self._predict_from_turns(conversation)

        self._load_model()

        audio, sr = sf.read(conversation.audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            import torchaudio
            import torch
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            tensor = resampler(tensor)
            audio = tensor.squeeze().numpy()
            sr = SAMPLE_RATE

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        events: list[PredictedEvent] = []

        for i in range(len(conversation.turns)):
            turn = conversation.turns[i]

            # --- Mid-turn probe (hold point) ---
            # At 50% through each turn, speaker is still talking → should be "hold"
            if turn.duration >= 1.0:
                mid_time = turn.start + turn.duration * 0.5
                mid_sample = int(mid_time * sr)
                mid_start = max(0, mid_sample - WINDOW_SAMPLES)

                if 0 < mid_sample <= len(audio):
                    window = audio[mid_start:mid_sample]
                    if len(window) >= sr:
                        result = self._predict_audio(window)
                        event_type = "shift" if result["prediction"] == 1 else "hold"
                        confidence = result["probability"] if event_type == "shift" else 1.0 - result["probability"]
                        events.append(PredictedEvent(
                            timestamp=mid_time,
                            event_type=event_type,
                            confidence=confidence,
                        ))

            # --- Turn boundary probe (shift point) ---
            if i == 0:
                continue
            boundary_time = turn.start
            end_sample = int(boundary_time * sr)
            start_sample = max(0, end_sample - WINDOW_SAMPLES)

            if end_sample <= 0 or end_sample > len(audio):
                continue

            window = audio[start_sample:end_sample]
            if len(window) < sr:
                continue

            result = self._predict_audio(window)
            event_type = "shift" if result["prediction"] == 1 else "hold"
            confidence = result["probability"] if event_type == "shift" else 1.0 - result["probability"]

            events.append(PredictedEvent(
                timestamp=boundary_time,
                event_type=event_type,
                confidence=confidence,
            ))

        return events

    def _predict_from_turns(self, conversation: Conversation) -> list[PredictedEvent]:
        """Fallback when no audio: always predict shift at boundaries."""
        events: list[PredictedEvent] = []
        for i in range(1, len(conversation.turns)):
            events.append(PredictedEvent(
                timestamp=conversation.turns[i].start,
                event_type="shift",
                confidence=0.5,
            ))
        return events
