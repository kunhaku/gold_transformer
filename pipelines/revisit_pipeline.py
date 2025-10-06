"""High-level orchestration that wires the revisit supervisor into training."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from config import DataConfig, ModelConfig
from decision import RevisitSupervisor
from data.datasets import SequenceDataset
from pipelines.training_pipeline import run_training_pipeline


PlanSpec = Mapping[str, Any]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _default_plan(dataset: SequenceDataset) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for group_id in np.unique(dataset.group_ids):
        plan.append(
            {
                "thesis_id": f"group_{int(group_id)}",
                "group_id": int(group_id),
                "open_step": 0,
                "revisit_steps": "all",
            }
        )
    return plan


def _normalise_plan(plan: Sequence[PlanSpec]) -> list[dict[str, Any]]:
    return [dict(spec) for spec in plan]


def _iter_revisit_specs(
    supervisor: RevisitSupervisor,
    group_id: int,
    open_step: int,
    spec: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    revisits = spec.get("revisits")
    if isinstance(revisits, Sequence):
        for entry in revisits:
            yield entry
        return

    steps = spec.get("revisit_steps")
    total_steps = supervisor.trace_length(group_id)

    if steps in (None, "all"):
        step_range: Iterable[Any] = range(open_step + 1, total_steps)
    elif isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        step_range = steps
    else:
        step_range = [steps]

    for step in step_range:
        yield {"step": int(step)}


def run_revisit_workflow(
    data_config: DataConfig | None = None,
    model_config: ModelConfig | None = None,
    *,
    revisit_plan: Sequence[PlanSpec] | None = None,
) -> dict[str, Any]:
    """Execute the full training + decision pipeline with revisit supervision."""

    training_result = run_training_pipeline(data_config, model_config)
    model = training_result["model"]
    datasets = training_result["datasets"]
    test_dataset: SequenceDataset = datasets["test"]

    plan = _normalise_plan(
        revisit_plan if revisit_plan is not None else _default_plan(test_dataset)
    )

    supervisor = RevisitSupervisor(
        model,
        test_dataset,
        scaler_metadata_path=training_result["artifacts"]["scaler_metadata"],
    )

    for spec in plan:
        thesis_id = spec["thesis_id"]
        group_id = int(spec["group_id"])
        open_step = int(spec.get("open_step", 0))

        thesis = supervisor.open_thesis(
            thesis_id,
            group_id,
            step=open_step,
            occurred_at=_parse_datetime(spec.get("open_occurred_at")),
            thesis_metadata=dict(spec.get("thesis_metadata") or {}),
            event_metadata=dict(spec.get("open_metadata") or {}),
        )

        for revisit in _iter_revisit_specs(supervisor, group_id, open_step, spec):
            step = revisit.get("step")
            supervisor.revisit_thesis(
                thesis.thesis_id,
                step=int(step) if step is not None else None,
                occurred_at=_parse_datetime(revisit.get("occurred_at")),
                metadata=dict(revisit.get("metadata") or {}),
            )

    thesis_summaries = {
        thesis.thesis_id: thesis.summary() for thesis in supervisor.list_theses()
    }

    training_result["decision"] = {
        "supervisor": supervisor,
        "plan": plan,
        "thesis_summaries": thesis_summaries,
    }

    return training_result

