from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from gbgad.data import load_dataset


@dataclass
class ExperimentDataset:
    """Feature/label bundle for experiments with labels excluded from X."""

    features: np.ndarray
    labels: np.ndarray | None
    feature_key: str
    label_key: str | None
    path: Path


def load_experiment_dataset(path: str | Path) -> ExperimentDataset:
    """Load MAT/NPZ/CSV data and keep labels exclusively for evaluation.

    In the benchmark MAT files, ``trandata`` stores the binary label in its
    final column.  The legacy GBGAD loader treats the whole matrix as features;
    this isolated loader follows the benchmark convention and removes that
    final label column without modifying any existing project module.
    """

    dataset_path = Path(path)
    bundle = load_dataset(dataset_path)
    features = bundle.features
    labels = bundle.labels
    feature_key = bundle.feature_key
    label_key = bundle.label_key

    # The legacy MAT benchmark stores labels in the final column of
    # ``trandata``.  Remove that column before training whenever it is binary.
    if dataset_path.suffix.lower() == ".mat" and feature_key == "trandata":
        raw_trandata = np.asarray(
            loadmat(dataset_path, variable_names=("trandata",))["trandata"]
        )
        if raw_trandata.ndim != 2 or raw_trandata.shape[1] < 2:
            raise ValueError(
                f"Expected trandata to be a two-dimensional matrix in {dataset_path}."
            )
        last_column = raw_trandata[:, -1].reshape(-1)
        if np.unique(last_column).size == 2:
            features = raw_trandata[:, :-1]
            labels = last_column
            feature_key = "trandata_without_label"
            label_key = "trandata_last_column"
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature matrix, got {features.shape}."
        )
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"Label count {labels.shape[0]} does not match N={features.shape[0]}."
            )
    return ExperimentDataset(
        features=features,
        labels=labels,
        feature_key=feature_key,
        label_key=label_key or None,
        path=dataset_path,
    )
