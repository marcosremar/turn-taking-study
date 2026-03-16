"""
LiveKit End-of-Turn (EOT) model benchmark.

Uses a fine-tuned Qwen2.5-0.5B model distilled from Qwen2.5-7B-Instruct
to predict end-of-turn from transcribed text.

Note: This model requires ASR transcription as input (text-based).

References:
- LiveKit. (2024). Using a Transformer to Improve End of Turn Detection.
  https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection
- LiveKit. (2025). Improved End-of-Turn Model Cuts Voice AI Interruptions 39%.
  https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/
- Qwen Team. (2024). Qwen2.5: A Party of Foundation Models.
  https://arxiv.org/abs/2412.15115
"""

from __future__ import annotations

import logging
import time

import numpy as np

from benchmark_base import TurnTakingModel, PredictedEvent
from setup_dataset import Conversation

log = logging.getLogger(__name__)


class LiveKitEOTModel(TurnTakingModel):
    """LiveKit End-of-Turn detection model (text-based, Qwen2.5-0.5B)."""

    def __init__(self, threshold: float = 0.5, device: str = "auto"):
        self.threshold = threshold
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "livekit_eot"

    @property
    def requires_gpu(self) -> bool:
        return False  # Designed for CPU inference

    @property
    def requires_asr(self) -> bool:
        return True  # Needs transcribed text

    def get_model_size_mb(self) -> float:
        return 281.0  # ~281MB on disk

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "livekit/turn-detector"
        log.info("Loading LiveKit turn-detector from %s", model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU-optimized
        )
        self._model.eval()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)

    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        self._load_model()
        events: list[PredictedEvent] = []

        # Build conversation context and evaluate at each turn boundary
        context_turns: list[dict[str, str]] = []

        for i, turn in enumerate(conversation.turns):
            if not turn.text or turn.text.startswith("[synthetic"):
                continue

            context_turns.append({
                "speaker": turn.speaker,
                "text": turn.text,
            })

            # Evaluate EOT probability after each turn
            eot_prob, latency = self._get_eot_probability(context_turns)

            if i < len(conversation.turns) - 1:
                next_turn = conversation.turns[i + 1]

                if eot_prob >= self.threshold:
                    events.append(PredictedEvent(
                        timestamp=turn.end,
                        event_type="shift",
                        confidence=eot_prob,
                        latency_ms=latency,
                    ))
                else:
                    events.append(PredictedEvent(
                        timestamp=turn.end,
                        event_type="hold",
                        confidence=1.0 - eot_prob,
                        latency_ms=latency,
                    ))

        return events

    def _get_eot_probability(self, turns: list[dict[str, str]]) -> tuple[float, float]:
        """
        Get end-of-turn probability for the current conversation state.

        Returns: (probability, latency_ms)
        """
        import torch

        # Format as chat-style prompt
        # LiveKit model expects conversation formatted with speaker tags
        prompt_parts = []
        for turn in turns[-5:]:  # Last 5 turns for context
            speaker_tag = "<|user|>" if turn["speaker"] in ("A", "caller") else "<|assistant|>"
            prompt_parts.append(f"{speaker_tag}\n{turn['text']}")

        prompt = "\n".join(prompt_parts)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Get probability of end-of-turn vs continuation
            probs = torch.softmax(logits, dim=-1)

            # Use EOS token probability as EOT signal
            eos_id = self._tokenizer.eos_token_id
            if eos_id is not None:
                eot_prob = float(probs[0, eos_id])
            else:
                # Fallback: use max prob as confidence proxy
                eot_prob = float(probs.max())

        latency = (time.perf_counter() - t0) * 1000.0

        return eot_prob, latency
