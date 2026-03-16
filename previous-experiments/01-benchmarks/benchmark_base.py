"""
Base classes and evaluation metrics for turn-taking benchmarks.

Metrics follow standard turn-taking evaluation methodology:
- Ekstedt, E. & Torre, G. (2024). Voice Activity Projection: Self-supervised
  Learning of Turn-taking Events. arXiv:2401.04868.
- Skantze, G. (2021). Turn-taking in Conversational Systems and Human-Robot
  Interaction: A Review. Computer Speech & Language, 67.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from setup_dataset import Conversation, TurnSegment

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class PredictedEvent:
    """A predicted turn-taking event."""
    timestamp: float  # seconds
    event_type: str  # "shift" or "hold"
    confidence: float = 1.0
    latency_ms: float = 0.0  # inference latency


@dataclass
class BenchmarkResult:
    """Results from evaluating a single model on the dataset."""
    model_name: str
    dataset_name: str
    # Classification metrics
    precision_shift: float = 0.0
    recall_shift: float = 0.0
    f1_shift: float = 0.0
    precision_hold: float = 0.0
    recall_hold: float = 0.0
    f1_hold: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    # Timing metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    # Turn-specific metrics
    mean_shift_delay_ms: float = 0.0  # How early/late shifts are detected
    false_interruption_rate: float = 0.0  # False positive shifts
    missed_shift_rate: float = 0.0  # False negative shifts
    # Resource usage
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    requires_gpu: bool = False
    requires_asr: bool = False
    # Metadata
    n_conversations: int = 0
    n_predictions: int = 0
    total_audio_hours: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class TurnTakingModel(ABC):
    """Abstract base for turn-taking prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        ...

    @property
    @abstractmethod
    def requires_asr(self) -> bool:
        ...

    @abstractmethod
    def predict(self, conversation: Conversation) -> list[PredictedEvent]:
        """Predict turn-taking events for a conversation."""
        ...

    def get_model_size_mb(self) -> float:
        """Return model size in MB."""
        return 0.0


def evaluate_model(
    model: TurnTakingModel,
    conversations: list[Conversation],
    dataset_name: str,
    tolerance_ms: float = 500.0,
) -> BenchmarkResult:
    """
    Evaluate a turn-taking model against ground truth annotations.

    Args:
        model: The model to evaluate
        conversations: List of conversations with ground truth
        dataset_name: Name of the dataset
        tolerance_ms: Matching tolerance in milliseconds for event alignment

    Returns:
        BenchmarkResult with all metrics computed
    """
    all_true_labels: list[int] = []
    all_pred_labels: list[int] = []
    all_latencies: list[float] = []
    shift_delays: list[float] = []
    false_interruptions = 0
    missed_shifts = 0
    total_shifts = 0
    total_predictions = 0

    tolerance_s = tolerance_ms / 1000.0

    for conv in conversations:
        t0 = time.perf_counter()
        predictions = model.predict(conv)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if predictions:
            per_pred_latency = elapsed_ms / len(predictions)
            all_latencies.extend([per_pred_latency] * len(predictions))
        total_predictions += len(predictions)

        # Build ground truth event timeline
        gt_shifts = set(conv.turn_shifts)
        gt_holds = set(conv.holds)
        total_shifts += len(gt_shifts)

        # Match predictions to ground truth events
        matched_shifts: set[float] = set()
        matched_holds: set[float] = set()

        for pred in predictions:
            matched = False

            # Check if prediction matches a ground truth shift
            for gt_t in gt_shifts:
                if abs(pred.timestamp - gt_t) <= tolerance_s:
                    if pred.event_type == "shift":
                        all_true_labels.append(1)
                        all_pred_labels.append(1)
                        matched_shifts.add(gt_t)
                        shift_delays.append((pred.timestamp - gt_t) * 1000.0)
                    else:
                        all_true_labels.append(1)
                        all_pred_labels.append(0)
                    matched = True
                    break

            if matched:
                continue

            # Check if prediction matches a ground truth hold
            for gt_t in gt_holds:
                if abs(pred.timestamp - gt_t) <= tolerance_s:
                    if pred.event_type == "hold":
                        all_true_labels.append(0)
                        all_pred_labels.append(0)
                        matched_holds.add(gt_t)
                    else:
                        all_true_labels.append(0)
                        all_pred_labels.append(1)
                        false_interruptions += 1
                    matched = True
                    break

            if not matched:
                # Unmatched prediction = false positive
                if pred.event_type == "shift":
                    all_true_labels.append(0)
                    all_pred_labels.append(1)
                    false_interruptions += 1
                else:
                    all_true_labels.append(0)
                    all_pred_labels.append(0)

        # Unmatched ground truth shifts = missed
        for gt_t in gt_shifts:
            if gt_t not in matched_shifts:
                all_true_labels.append(1)
                all_pred_labels.append(0)
                missed_shifts += 1

    # Compute metrics
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)

    result = BenchmarkResult(
        model_name=model.name,
        dataset_name=dataset_name,
        n_conversations=len(conversations),
        n_predictions=total_predictions,
        total_audio_hours=sum(c.duration for c in conversations) / 3600.0,
        requires_gpu=model.requires_gpu,
        requires_asr=model.requires_asr,
        model_size_mb=model.get_model_size_mb(),
    )

    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        result.precision_shift = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        result.recall_shift = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        result.f1_shift = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        result.precision_hold = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        result.recall_hold = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        result.f1_hold = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        result.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if all_latencies:
        arr = np.array(all_latencies)
        result.mean_latency_ms = float(np.mean(arr))
        result.p50_latency_ms = float(np.percentile(arr, 50))
        result.p95_latency_ms = float(np.percentile(arr, 95))
        result.p99_latency_ms = float(np.percentile(arr, 99))

    if shift_delays:
        result.mean_shift_delay_ms = float(np.mean(shift_delays))

    if total_shifts > 0:
        result.missed_shift_rate = missed_shifts / total_shifts

    total_non_shifts = len(all_true_labels) - total_shifts
    if total_non_shifts > 0:
        result.false_interruption_rate = false_interruptions / total_non_shifts

    return result


def save_result(result: BenchmarkResult) -> Path:
    """Save benchmark result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result.model_name}_{result.dataset_name}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Saved result to %s", path)
    return path


def load_all_results() -> list[BenchmarkResult]:
    """Load all saved benchmark results."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        results.append(BenchmarkResult(**data))
    return results
