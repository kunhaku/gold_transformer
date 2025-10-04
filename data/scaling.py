"""Utilities for persisting and applying feature scaling metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

ScalerParams = Mapping[str, Any]


def save_scaler_metadata(
    path: str | Path,
    feature_metadata: Mapping[str, Mapping[str, float]],
    *,
    target_metadata: Mapping[str, float] | None = None,
    target_column: str = "close",
) -> None:
    """Persist scaler metadata for model features and targets.

    Parameters
    ----------
    path:
        Destination path for the JSON artifact.
    feature_metadata:
        Mapping of feature name to the scaler parameters that were used when
        transforming the inputs. The values are expected to contain scalar
        statistics (e.g. ``{"mean": 0.0, "scale": 1.0}``).
    target_metadata:
        Optional metadata for the prediction target. When provided, the target
        statistics are stored under the ``target`` key so downstream
        components can apply an inverse transform.
    target_column:
        Name of the target column. Defaults to ``"close"`` to align with the
        primary forecasting objective of the project.
    """

    payload: MutableMapping[str, Any] = {"features": dict(feature_metadata)}
    if target_metadata is not None:
        payload["target"] = {target_column: dict(target_metadata)}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scaler_metadata(path: str | Path | None) -> Optional[dict[str, Any]]:
    """Load scaler metadata from *path* if it exists.

    The function returns ``None`` when *path* is ``None`` or when the file is
    missing. This behaviour lets callers fall back to scaled values when no
    metadata was produced alongside the dataset artifacts.
    """

    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_named_metadata(container: Any, column: str) -> Optional[ScalerParams]:
    """Return metadata for *column* from *container* if available.

    The helper accepts a variety of shapes for backwards compatibility:

    - ``{"close": {"mean": 0.0, "scale": 1.0}}``
    - ``[{"name": "close", "mean": 0.0, "scale": 1.0}]``
    - ``{"name": "close", "stats": {...}}``
    """

    if container is None:
        return None

    if isinstance(container, Mapping):
        if column in container and isinstance(container[column], Mapping):
            return container[column]  # direct mapping

        name = container.get("name") or container.get("column")
        if name == column:
            stats = container.get("stats")
            if isinstance(stats, Mapping):
                return stats
            if isinstance(container, Mapping):
                return container

        # Some serialisations wrap stats under a "columns" key.
        columns = container.get("columns")
        if isinstance(columns, Mapping):
            return _get_named_metadata(columns, column)

    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for item in container:
            result = _get_named_metadata(item, column)
            if result is not None:
                return result

    return None


def get_feature_scaler(metadata: Mapping[str, Any] | None, column: str) -> Optional[ScalerParams]:
    """Extract scaler parameters for an input *column* from *metadata*."""

    if not metadata:
        return None
    features = metadata.get("features")
    return _get_named_metadata(features, column)


def get_target_scaler(
    metadata: Mapping[str, Any] | None,
    target_column: str = "close",
) -> Optional[ScalerParams]:
    """Extract scaler parameters for the target column from *metadata*."""

    if not metadata:
        return None

    target_section = metadata.get("target")
    if target_section is None:
        return None
    return _get_named_metadata(target_section, target_column)


def inverse_transform(values: np.ndarray | Sequence[float], scaler: ScalerParams) -> np.ndarray:
    """Apply an inverse transformation defined by *scaler* to *values*.

    The helper supports a handful of scaler parameter conventions commonly
    produced by scikit-learn classes:

    - Standard/Robust scaler: ``{"mean": m, "scale": s}`` or
      ``{"center": m, "scale": s}``
    - Min-max scaler: ``{"min": a, "max": b}``
    - Custom mappings with ``offset``/``scale`` pairs.
    """

    arr = np.asarray(values, dtype=np.float32)
    arr = arr.copy()

    if scaler is None:
        return arr

    def _as_array(key: str) -> Optional[np.ndarray]:
        if key not in scaler:
            return None
        value = scaler[key]
        return np.asarray(value, dtype=np.float32)

    scale = _as_array("scale") or _as_array("std") or _as_array("var")
    offset = _as_array("mean") or _as_array("center") or _as_array("median")
    minimum = _as_array("min") or _as_array("data_min")
    maximum = _as_array("max") or _as_array("data_max")

    if minimum is not None and maximum is not None:
        # Assume data were scaled to [0, 1]. Support broadcasting when
        # metadata was stored as scalars.
        denom = maximum - minimum
        denom = np.where(denom == 0, 1.0, denom)
        arr = arr * denom + minimum
        return arr

    if scale is not None and offset is not None:
        arr = arr * scale + offset
        return arr

    if scale is not None:
        arr = arr * scale

    if offset is not None:
        arr = arr + offset

    if "add" in scaler:
        arr = arr + np.asarray(scaler["add"], dtype=np.float32)
    if "mul" in scaler:
        arr = arr * np.asarray(scaler["mul"], dtype=np.float32)

    return arr


def is_probably_scaled(values: Sequence[float], scaler: ScalerParams | None) -> bool:
    """Heuristic to detect whether *values* are likely in scaled units.

    The heuristic compares the typical magnitude of *values* with the stored
    scaler statistics. It errs on the side of returning ``False`` (treating
    the values as already in human-readable units) when insufficient
    information is available.
    """

    if scaler is None:
        return False

    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False

    minimum = scaler.get("min") or scaler.get("data_min")
    maximum = scaler.get("max") or scaler.get("data_max")
    if minimum is not None and maximum is not None:
        try:
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
        except ValueError:
            return False
        # Min-max scaled data typically lives within [0, 1]. Allow for a bit
        # of numerical noise.
        if min_val >= -0.1 and max_val <= 1.1:
            return True
        return False

    reference = scaler.get("mean") or scaler.get("center") or scaler.get("median")
    dispersion = scaler.get("scale") or scaler.get("std") or scaler.get("var")

    if reference is not None and dispersion is not None:
        reference = float(np.asarray(reference).reshape(-1)[0])
        dispersion = float(np.asarray(dispersion).reshape(-1)[0])
        median_distance = float(np.median(np.abs(arr - reference)))
        # When the data are scaled (centred around zero), their distance from
        # the original reference is typically many multiples of the scaling
        # factor. Conversely, unscaled data hover within a few multiples.
        return median_distance > dispersion * 4

    # Fallback: small absolute magnitudes hint at scaled values.
    return float(np.median(np.abs(arr))) < 5
