"""Supervisory utilities implementing the revisit hierarchy.

The classes in this module layer a decision process above the base
transformer forecaster described in :mod:`docs/model_diagnostics`. They let
callers open trade theses, request revisits, and compare the resulting
forecasts without polluting the supervised training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from data.datasets import SequenceDataset
from data.scaling import load_scaler_metadata
from evaluation import compute_regression_metrics
from evaluation.inference import ForecastRecord, generate_group_forecast


@dataclass(frozen=True)
class ForecastSnapshot:
    """Immutable view of a single forecast/target pair."""

    group_id: int
    dataset_index: int
    step: int
    valid_length: int
    prediction: np.ndarray
    target: Optional[np.ndarray]
    metrics: Dict[str, float]


@dataclass
class ThesisEvent:
    """A timestamped forecast that belongs to a trade thesis."""

    snapshot: ForecastSnapshot
    occurred_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeThesis:
    """State container tracking the full revisit history for a thesis."""

    thesis_id: str
    group_id: int
    open_event: ThesisEvent
    metadata: Dict[str, Any] = field(default_factory=dict)
    revisits: List[ThesisEvent] = field(default_factory=list)

    def add_revisit(self, event: ThesisEvent) -> None:
        self.revisits.append(event)

    def latest_event(self) -> ThesisEvent:
        return self.revisits[-1] if self.revisits else self.open_event

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable view of the thesis and its revisits."""

        base_snapshot = self.open_event.snapshot
        base_prediction = base_snapshot.prediction

        def _event_payload(event: ThesisEvent) -> Dict[str, Any]:
            snapshot = event.snapshot
            payload: Dict[str, Any] = {
                "step": snapshot.step,
                "dataset_index": snapshot.dataset_index,
                "valid_length": snapshot.valid_length,
                "prediction": snapshot.prediction.tolist(),
                "target": snapshot.target.tolist() if snapshot.target is not None else None,
                "metrics": snapshot.metrics,
                "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
                "metadata": event.metadata,
            }
            if snapshot.prediction is not None and base_prediction is not None:
                payload["delta_vs_initial"] = (
                    snapshot.prediction - base_prediction
                ).tolist()
            return payload

        return {
            "thesis_id": self.thesis_id,
            "group_id": self.group_id,
            "metadata": self.metadata,
            "opened_at": self.open_event.occurred_at.isoformat()
            if self.open_event.occurred_at
            else None,
            "initial": _event_payload(self.open_event),
            "revisits": [_event_payload(evt) for evt in self.revisits],
        }


class RevisitSupervisor:
    """Decision-layer coordinator that manages revisit workflows."""

    def __init__(
        self,
        model: Any,
        dataset: SequenceDataset,
        *,
        scaler_metadata: Optional[Mapping[str, Any]] = None,
        scaler_metadata_path: Optional[str] = None,
    ) -> None:
        if scaler_metadata is None and scaler_metadata_path is not None:
            scaler_metadata = load_scaler_metadata(scaler_metadata_path)

        self._model = model
        self._dataset = dataset
        self._scaler_metadata: Optional[Mapping[str, Any]] = scaler_metadata
        self._group_indices: Dict[int, np.ndarray] = {
            int(group_id): np.where(dataset.group_ids == group_id)[0]
            for group_id in np.unique(dataset.group_ids)
        }
        self._trace_cache: Dict[int, List[ForecastSnapshot]] = {}
        self._theses: Dict[str, TradeThesis] = {}

    # ------------------------------------------------------------------
    # Forecast helpers
    # ------------------------------------------------------------------
    def _build_snapshot(self, record: ForecastRecord, group_indices: np.ndarray) -> ForecastSnapshot:
        dataset_index = int(group_indices[record.step]) if record.step < len(group_indices) else int(record.sample_idx)
        metrics: Dict[str, float] = {}
        if record.target is not None:
            metrics = compute_regression_metrics(
                record.target.reshape(1, -1), record.prediction.reshape(1, -1)
            )
        return ForecastSnapshot(
            group_id=record.group_id,
            dataset_index=dataset_index,
            step=record.step,
            valid_length=record.valid_length,
            prediction=record.prediction,
            target=record.target,
            metrics=metrics,
        )

    def _get_trace(self, group_id: int) -> List[ForecastSnapshot]:
        if group_id not in self._trace_cache:
            if group_id not in self._group_indices:
                raise KeyError(f"Unknown group_id {group_id}")
            records = generate_group_forecast(
                self._model,
                self._dataset,
                group_id,
                scaler_metadata=self._scaler_metadata,
            )
            indices = self._group_indices[group_id]
            snapshots = [self._build_snapshot(record, indices) for record in records]
            self._trace_cache[group_id] = snapshots
        return self._trace_cache[group_id]

    def _clone_snapshot(
        self,
        snapshot: ForecastSnapshot,
        *,
        occurred_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ThesisEvent:
        return ThesisEvent(
            snapshot=snapshot,
            occurred_at=occurred_at,
            metadata=dict(metadata or {}),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open_thesis(
        self,
        thesis_id: str,
        group_id: int,
        *,
        step: int = 0,
        occurred_at: Optional[datetime] = None,
        thesis_metadata: Optional[Mapping[str, Any]] = None,
        event_metadata: Optional[Mapping[str, Any]] = None,
    ) -> TradeThesis:
        if thesis_id in self._theses:
            raise ValueError(f"Thesis '{thesis_id}' already exists")

        trace = self._get_trace(group_id)
        if step >= len(trace):
            raise IndexError(
                f"Group {group_id} only has {len(trace)} steps; requested step {step}"
            )

        open_event = self._clone_snapshot(
            trace[step], occurred_at=occurred_at, metadata=event_metadata
        )
        thesis = TradeThesis(
            thesis_id=thesis_id,
            group_id=group_id,
            open_event=open_event,
            metadata=dict(thesis_metadata or {}),
        )
        self._theses[thesis_id] = thesis
        return thesis

    def revisit_thesis(
        self,
        thesis_id: str,
        *,
        step: Optional[int] = None,
        occurred_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ThesisEvent:
        thesis = self.get_thesis(thesis_id)
        trace = self._get_trace(thesis.group_id)

        if step is None:
            step = thesis.latest_event().snapshot.step + 1

        if step >= len(trace):
            raise IndexError(
                f"Group {thesis.group_id} only has {len(trace)} steps; requested step {step}"
            )

        event = self._clone_snapshot(trace[step], occurred_at=occurred_at, metadata=metadata)
        thesis.add_revisit(event)
        return event

    def get_thesis(self, thesis_id: str) -> TradeThesis:
        if thesis_id not in self._theses:
            raise KeyError(f"Unknown thesis '{thesis_id}'")
        return self._theses[thesis_id]

    def summarize_thesis(self, thesis_id: str) -> Dict[str, Any]:
        return self.get_thesis(thesis_id).summary()

    def list_theses(self) -> List[TradeThesis]:
        return list(self._theses.values())

    def reset_cache(self) -> None:
        """Clear cached traces, forcing recomputation on next access."""

        self._trace_cache.clear()
