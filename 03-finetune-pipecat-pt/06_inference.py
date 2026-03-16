"""Inference engine for the language-learning avatar turn-taking model.

This module implements the dual-threshold + backchannel system designed
for a conversational avatar that teaches Portuguese to French speakers.

Key insight: L2 learners pause 1-3s mid-sentence (word search, conjugation,
code-switching). A naive system either:
  a) Interrupts them (bad UX — kills confidence), or
  b) Stays silent too long (learner thinks avatar froze)

Solution: dual-threshold with backchannel signals.

Architecture:
    Audio stream → SmartTurnV3 model → confidence score (0.0-1.0)
        │
        ├─ score < eager_threshold (0.4)  → LISTENING (do nothing)
        ├─ eager ≤ score < final (0.7)    → PREPARING (backchannel + speculative LLM)
        └─ score ≥ final_threshold (0.7)  → RESPONDING (take the turn)

    Silence timer (parallel):
        0-600ms    → normal (no action)
        600ms-1.5s → visual backchannel (nod, eye contact)
        1.5s-3.0s  → verbal backchannel ("mhm", "continue...")
        3.0s+      → encouragement ("sem pressa, pode pensar...")

References:
- Deepgram Flux: eot_threshold + eager_eot_threshold (2025)
- ConversAR: "infinite thinking period" for L2 learners (2025)
- Tavus: 600ms response latency threshold (2025)
- Kosmala (2022): L2 repair pauses average 844ms
- Cenoz (2000): L2 silent pauses range 205ms to 11,569ms

Usage:
    engine = TurnTakingEngine(model_path="results/smart_turn_pt_v3.onnx")
    engine.on_state_change(my_callback)

    # Feed audio chunks continuously
    for chunk in audio_stream:
        engine.feed_audio(chunk)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE


# ---------------------------------------------------------------------------
# States and events
# ---------------------------------------------------------------------------

class TurnState(Enum):
    """Avatar turn-taking state machine."""
    LISTENING = "listening"           # User is speaking, avatar listens
    SILENCE = "silence"               # User stopped, waiting to classify
    BACKCHANNEL_VISUAL = "bc_visual"  # Nod/eye contact (600ms-1.5s silence)
    BACKCHANNEL_VERBAL = "bc_verbal"  # "Mhm" / "continue" (1.5s-3.0s)
    ENCOURAGEMENT = "encouragement"   # "Sem pressa..." (3.0s+ silence)
    PREPARING = "preparing"           # Eager threshold hit, speculative LLM started
    RESPONDING = "responding"         # Final threshold hit, avatar speaks


@dataclass
class TurnEvent:
    """Event emitted on state transitions."""
    state: TurnState
    confidence: float          # Model's end-of-turn confidence (0-1)
    silence_duration_ms: float # How long the user has been silent
    timestamp: float           # time.monotonic()
    message: str = ""          # Backchannel text (if applicable)


@dataclass
class BackchannelConfig:
    """Configurable backchannel behavior for the avatar.

    These can be tuned per learner profile or CEFR level:
    - A1/A2: longer delays, more encouraging messages
    - B1/B2: shorter delays, less frequent backchannels
    """
    # Silence thresholds (milliseconds)
    visual_backchannel_ms: float = 600.0
    verbal_backchannel_ms: float = 1500.0
    encouragement_ms: float = 3000.0

    # Model confidence thresholds
    eager_threshold: float = 0.4    # Start speculative LLM generation
    final_threshold: float = 0.7    # Confirm end-of-turn, avatar speaks

    # Backchannel messages (Portuguese, with French-speaker-friendly variants)
    visual_actions: list[str] = field(default_factory=lambda: [
        "nod",          # Aceno de cabeça
        "eye_contact",  # Olhar atento
        "slight_smile", # Sorriso leve
    ])

    verbal_backchannels: list[str] = field(default_factory=lambda: [
        "Mhm...",
        "Uhum...",
        "Sim...",
        "Tá...",
        "Sei...",
        "Continue...",
    ])

    encouragement_messages: list[str] = field(default_factory=lambda: [
        "Pode continuar, sem pressa...",
        "Tá pensando? Tranquilo...",
        "Pode pensar, eu espero...",
        "Sem pressa, tá tudo bem...",
        "Take your time... pode falar em português...",  # Code-switch friendly
        "Prenez votre temps... quando estiver pronto...",  # French reassurance
    ])

    # Cooldowns (don't spam backchannels)
    verbal_cooldown_ms: float = 3000.0   # Min time between verbal backchannels
    encouragement_cooldown_ms: float = 8000.0  # Min time between encouragements


# ---------------------------------------------------------------------------
# CEFR-aware presets
# ---------------------------------------------------------------------------

CEFR_PRESETS: dict[str, BackchannelConfig] = {
    "A1": BackchannelConfig(
        visual_backchannel_ms=500.0,
        verbal_backchannel_ms=1200.0,
        encouragement_ms=2500.0,
        eager_threshold=0.5,     # Higher — need more confidence before preparing
        final_threshold=0.8,     # Much higher — very patient, rarely interrupt
    ),
    "A2": BackchannelConfig(
        visual_backchannel_ms=550.0,
        verbal_backchannel_ms=1300.0,
        encouragement_ms=2800.0,
        eager_threshold=0.45,
        final_threshold=0.75,
    ),
    "B1": BackchannelConfig(
        visual_backchannel_ms=600.0,
        verbal_backchannel_ms=1500.0,
        encouragement_ms=3000.0,
        eager_threshold=0.4,
        final_threshold=0.7,     # Default
    ),
    "B2": BackchannelConfig(
        visual_backchannel_ms=600.0,
        verbal_backchannel_ms=1800.0,
        encouragement_ms=4000.0,
        eager_threshold=0.35,
        final_threshold=0.65,    # More responsive — B2 pauses less
    ),
    "C1": BackchannelConfig(
        visual_backchannel_ms=600.0,
        verbal_backchannel_ms=2000.0,
        encouragement_ms=5000.0,
        eager_threshold=0.35,
        final_threshold=0.6,     # Near-native responsiveness
    ),
}


# ---------------------------------------------------------------------------
# Turn-taking engine
# ---------------------------------------------------------------------------

class TurnTakingEngine:
    """Real-time turn-taking engine for language-learning avatar.

    Combines the SmartTurnV3 model with a silence timer and backchannel
    state machine to create a patient, encouraging conversational partner.

    Usage:
        engine = TurnTakingEngine("results/smart_turn_pt_v3.onnx")
        engine.on_state_change(handle_event)

        # In audio loop:
        for chunk in mic_stream:
            engine.feed_audio(chunk)

        # Or feed pre-computed features:
        engine.feed_score(model_confidence, is_speech=True)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: BackchannelConfig | None = None,
        cefr_level: str = "B1",
    ):
        # Config: use CEFR preset or custom
        if config is not None:
            self.config = config
        elif cefr_level in CEFR_PRESETS:
            self.config = CEFR_PRESETS[cefr_level]
            log.info("Using CEFR %s preset: eager=%.2f, final=%.2f",
                     cefr_level, self.config.eager_threshold, self.config.final_threshold)
        else:
            self.config = BackchannelConfig()

        # State
        self._state = TurnState.LISTENING
        self._silence_start: float | None = None
        self._last_speech_time: float = time.monotonic()
        self._last_verbal_bc: float = 0.0
        self._last_encouragement: float = 0.0
        self._last_confidence: float = 0.0
        self._callbacks: list[Callable[[TurnEvent], None]] = []
        self._audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._speculative_started = False

        # Load ONNX model if provided
        self._session = None
        self._feature_extractor = None
        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str | Path) -> None:
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            log.info("Loaded turn-taking model: %s", model_path)
        except Exception as e:
            log.warning("Failed to load model %s: %s — running in manual mode", model_path, e)

        try:
            from transformers import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)
        except ImportError:
            log.warning("transformers not available — cannot extract features")

    def on_state_change(self, callback: Callable[[TurnEvent], None]) -> None:
        """Register callback for state change events."""
        self._callbacks.append(callback)

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def silence_duration_ms(self) -> float:
        if self._silence_start is None:
            return 0.0
        return (time.monotonic() - self._silence_start) * 1000

    # ----- Audio input -----

    def feed_audio(self, chunk: np.ndarray, is_speech: bool | None = None) -> TurnEvent | None:
        """Feed an audio chunk and get turn-taking decision.

        chunk: float32 audio at 16kHz
        is_speech: if None, will use simple energy-based VAD

        Returns TurnEvent if state changed, None otherwise.
        """
        now = time.monotonic()

        # Update audio buffer (sliding window)
        chunk = chunk.astype(np.float32)
        if len(chunk) >= WINDOW_SAMPLES:
            self._audio_buffer = chunk[-WINDOW_SAMPLES:]
        else:
            self._audio_buffer = np.roll(self._audio_buffer, -len(chunk))
            self._audio_buffer[-len(chunk):] = chunk

        # Simple VAD if not provided
        if is_speech is None:
            rms = np.sqrt(np.mean(chunk ** 2))
            is_speech = rms > 0.01  # Simple threshold

        if is_speech:
            return self._on_speech(now)
        else:
            return self._on_silence(now)

    def feed_score(self, confidence: float, is_speech: bool) -> TurnEvent | None:
        """Feed pre-computed model confidence score.

        Use this when you run the model externally and just want
        the backchannel/threshold logic.

        confidence: 0.0 (definitely incomplete) to 1.0 (definitely complete)
        is_speech: whether VAD detected speech in this frame
        """
        now = time.monotonic()
        self._last_confidence = confidence

        if is_speech:
            return self._on_speech(now)
        else:
            return self._on_silence(now, confidence=confidence)

    # ----- Internal state machine -----

    def _on_speech(self, now: float) -> TurnEvent | None:
        """User is speaking."""
        self._last_speech_time = now
        self._silence_start = None
        self._speculative_started = False

        if self._state != TurnState.LISTENING:
            return self._transition(TurnState.LISTENING, confidence=0.0, now=now)
        return None

    def _on_silence(self, now: float, confidence: float | None = None) -> TurnEvent | None:
        """User is silent — run the state machine."""
        # Mark silence start
        if self._silence_start is None:
            self._silence_start = now

        silence_ms = (now - self._silence_start) * 1000

        # Get model confidence if we have a model and no external score
        if confidence is None:
            confidence = self._run_model()
        self._last_confidence = confidence

        cfg = self.config

        # ----- Decision tree -----

        # 1. Final threshold → RESPONDING (take the turn)
        if confidence >= cfg.final_threshold:
            return self._transition(TurnState.RESPONDING, confidence, now)

        # 2. Eager threshold → PREPARING (speculative LLM, backchannel)
        if confidence >= cfg.eager_threshold and not self._speculative_started:
            self._speculative_started = True
            return self._transition(TurnState.PREPARING, confidence, now)

        # 3. Silence-based backchannels (even if model is unsure)
        # These keep the learner engaged so they don't think the avatar froze

        if silence_ms >= cfg.encouragement_ms:
            if now - self._last_encouragement >= cfg.encouragement_cooldown_ms / 1000:
                self._last_encouragement = now
                return self._transition(TurnState.ENCOURAGEMENT, confidence, now)

        if silence_ms >= cfg.verbal_backchannel_ms:
            if now - self._last_verbal_bc >= cfg.verbal_cooldown_ms / 1000:
                self._last_verbal_bc = now
                return self._transition(TurnState.BACKCHANNEL_VERBAL, confidence, now)

        if silence_ms >= cfg.visual_backchannel_ms:
            if self._state not in (TurnState.BACKCHANNEL_VISUAL,
                                    TurnState.BACKCHANNEL_VERBAL,
                                    TurnState.ENCOURAGEMENT,
                                    TurnState.PREPARING):
                return self._transition(TurnState.BACKCHANNEL_VISUAL, confidence, now)

        # Still in silence, no state change
        if self._state == TurnState.LISTENING:
            return self._transition(TurnState.SILENCE, confidence, now)

        return None

    def _transition(self, new_state: TurnState, confidence: float, now: float) -> TurnEvent:
        """Transition to a new state and emit event."""
        import random

        old_state = self._state
        self._state = new_state

        silence_ms = (now - self._silence_start) * 1000 if self._silence_start else 0.0

        # Pick appropriate message
        message = ""
        if new_state == TurnState.BACKCHANNEL_VISUAL:
            message = random.choice(self.config.visual_actions)
        elif new_state == TurnState.BACKCHANNEL_VERBAL:
            message = random.choice(self.config.verbal_backchannels)
        elif new_state == TurnState.ENCOURAGEMENT:
            message = random.choice(self.config.encouragement_messages)
        elif new_state == TurnState.PREPARING:
            message = "speculative_llm_start"
        elif new_state == TurnState.RESPONDING:
            message = "take_turn"

        event = TurnEvent(
            state=new_state,
            confidence=confidence,
            silence_duration_ms=silence_ms,
            timestamp=now,
            message=message,
        )

        if old_state != new_state:
            log.debug("Turn state: %s → %s (conf=%.2f, silence=%.0fms, msg=%s)",
                      old_state.value, new_state.value, confidence, silence_ms, message)

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                log.warning("Callback error: %s", e)

        return event

    def _run_model(self) -> float:
        """Run the ONNX model on current audio buffer."""
        if self._session is None or self._feature_extractor is None:
            return 0.0

        try:
            inputs = self._feature_extractor(
                self._audio_buffer,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=WINDOW_SAMPLES,
                truncation=True,
                do_normalize=True,
            )
            features = inputs.input_features.astype(np.float32)

            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: features})
            logit = outputs[0].item() if outputs[0].size == 1 else outputs[0][0].item()

            # Sigmoid
            confidence = 1.0 / (1.0 + np.exp(-logit))
            return float(confidence)
        except Exception as e:
            log.warning("Model inference error: %s", e)
            return 0.0

    # ----- Convenience -----

    def reset(self) -> None:
        """Reset state (e.g., when starting a new conversation turn)."""
        self._state = TurnState.LISTENING
        self._silence_start = None
        self._speculative_started = False
        self._last_confidence = 0.0
        self._audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    def set_cefr_level(self, level: str) -> None:
        """Change CEFR level at runtime (e.g., after proficiency assessment)."""
        if level in CEFR_PRESETS:
            self.config = CEFR_PRESETS[level]
            log.info("Switched to CEFR %s: eager=%.2f, final=%.2f",
                     level, self.config.eager_threshold, self.config.final_threshold)
        else:
            log.warning("Unknown CEFR level: %s", level)


# ---------------------------------------------------------------------------
# Example usage and demo
# ---------------------------------------------------------------------------

def demo_simulation():
    """Simulate a conversation to demonstrate the backchannel system.

    Scenario: French speaker (B1) learning Portuguese, pausing frequently.
    """
    print("=" * 70)
    print("DEMO: Turn-Taking Engine for Language Learning Avatar")
    print("Scenario: French B1 learner speaking Portuguese")
    print("=" * 70)

    events_log = []

    def on_event(event: TurnEvent):
        events_log.append(event)
        state_emoji = {
            TurnState.LISTENING: "👂",
            TurnState.SILENCE: "⏸️",
            TurnState.BACKCHANNEL_VISUAL: "😊",
            TurnState.BACKCHANNEL_VERBAL: "💬",
            TurnState.ENCOURAGEMENT: "🤗",
            TurnState.PREPARING: "🧠",
            TurnState.RESPONDING: "🗣️",
        }
        emoji = state_emoji.get(event.state, "❓")
        print(f"  {emoji} [{event.silence_duration_ms:6.0f}ms] "
              f"conf={event.confidence:.2f} → {event.state.value}: {event.message}")

    engine = TurnTakingEngine(config=CEFR_PRESETS["B1"])
    engine.on_state_change(on_event)

    # Simulate a conversation timeline
    # Each entry: (description, duration_ms, is_speech, model_confidence)
    timeline = [
        # Learner starts speaking
        ("Learner: 'Eu fui ao...'", 1200, True, 0.1),
        # Pause — searching for word (conjugation hesitation)
        ("  [silence — thinking about conjugation]", 300, False, 0.15),
        ("  [still thinking...]", 400, False, 0.20),
        ("  [600ms — visual backchannel]", 300, False, 0.22),
        ("  [1s — still silent]", 400, False, 0.25),
        ("  [1.5s — verbal backchannel]", 500, False, 0.28),
        # Learner continues
        ("Learner: '...mercado... euh...'", 800, True, 0.1),
        # Another pause — code-switching hesitation
        ("  [silence — 'comment dit-on...']", 500, False, 0.20),
        ("  [1s silence]", 500, False, 0.30),
        ("  [1.5s — verbal backchannel]", 500, False, 0.35),
        ("  [2s — model getting uncertain]", 500, False, 0.42),
        # Learner continues again
        ("Learner: '...para comprar... uma tesoura.'", 1500, True, 0.1),
        # Final silence — this time it's a real end of turn
        ("  [silence after complete sentence]", 300, False, 0.45),
        ("  [600ms]", 300, False, 0.55),
        ("  [model confidence rising]", 300, False, 0.65),
        ("  [model confident — end of turn]", 200, False, 0.75),
    ]

    print("\nTimeline:")
    print("-" * 70)

    for description, duration_ms, is_speech, confidence in timeline:
        print(f"\n{description} ({duration_ms}ms)")
        # Simulate in 100ms chunks
        n_chunks = max(1, duration_ms // 100)
        for _ in range(n_chunks):
            engine.feed_score(confidence, is_speech)
            time.sleep(0.001)  # Tiny sleep for monotonic clock to advance

    print("\n" + "=" * 70)
    print(f"Total events: {len(events_log)}")
    print(f"Final state: {engine.state.value}")

    # Show CEFR comparison
    print("\n" + "=" * 70)
    print("CEFR LEVEL COMPARISON")
    print("=" * 70)
    print(f"{'Level':<6} {'Eager':<8} {'Final':<8} {'Visual BC':<10} {'Verbal BC':<10} {'Encourage':<10}")
    print("-" * 52)
    for level, preset in CEFR_PRESETS.items():
        print(f"{level:<6} {preset.eager_threshold:<8.2f} {preset.final_threshold:<8.2f} "
              f"{preset.visual_backchannel_ms:<10.0f} {preset.verbal_backchannel_ms:<10.0f} "
              f"{preset.encouragement_ms:<10.0f}")
    print()
    print("A1/A2: Very patient — high final threshold (0.75-0.80), early encouragement")
    print("B1:    Default — balanced patience and responsiveness")
    print("B2/C1: More responsive — lower final threshold (0.60-0.65)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    demo_simulation()
