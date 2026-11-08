from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FEATURE_KEYS = ("X", "x", "fea", "features", "data", "samples")
LABEL_KEYS = ("y", "Y", "gnd", "label", "labels", "target", "targets")


@dataclass
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray | None
    feature_key: str
    label_key: str | None
    path: Path


def _normalize_matrix(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {matrix.shape}.")
    if matrix.shape[0] == 1 and matrix.shape[1] > 1:
        matrix = matrix.T
    return matrix.astype(np.float32)


def _normalize_labels(array: np.ndarray | None, sample_count: int) -> np.ndarray | None:
    if array is None:
        return None
    labels = np.asarray(array).reshape(-1)
    if labels.size != sample_count:
        if labels.size == 1:
            labels = np.repeat(labels, sample_count)
        else:
            raise ValueError(f"Label size {labels.size} does not match sample count {sample_count}.")
    return labels.astype(np.int64)


def _find_first_key(payload: dict[str, np.ndarray], candidates: tuple[str, ...]) -> tuple[str | None, np.ndarray | None]:
    for key in candidates:
        if key in payload:
            return key, payload[key]
    return None, None


def _load_generic_mapping_dataset(payload: dict[str, np.ndarray], dataset_path: Path) -> DatasetBundle:
    feature_key, feature_value = _find_first_key(payload, FEATURE_KEYS)
    label_key, label_value = _find_first_key(payload, LABEL_KEYS)

    if feature_value is None:
        numeric_candidates: list[tuple[str, np.ndarray]] = []
        for key, value in payload.items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number):
                numeric_candidates.append((key, array))
        if not numeric_candidates:
            raise ValueError(f"No numeric matrix found in {dataset_path}.")
        numeric_candidates.sort(key=lambda item: np.asarray(item[1]).size, reverse=True)
        feature_key, feature_value = numeric_candidates[0]

    features = _normalize_matrix(feature_value)

    if label_value is None:
        for key, value in payload.items():
            array = np.asarray(value)
            if key == feature_key or not np.issubdtype(array.dtype, np.number):
                continue
            if array.size == features.shape[0]:
                label_key = key
                label_value = array
                break

    labels = _normalize_labels(label_value, features.shape[0])
    return DatasetBundle(
        features=features,
        labels=labels,
        feature_key=feature_key or "unknown",
        label_key=label_key,
        path=dataset_path,
    )


def load_mat_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    payload = loadmat(dataset_path)
    visible_items = {k: v for k, v in payload.items() if not k.startswith("__")}
    return _load_generic_mapping_dataset(visible_items, dataset_path)


def load_csv_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    features = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32)
    features = _normalize_matrix(features)
    return DatasetBundle(
        features=features,
        labels=None,
        feature_key="csv",
        label_key=None,
        path=dataset_path,
    )


def load_npz_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        visible_items = {key: payload[key] for key in payload.files}
    return _load_generic_mapping_dataset(visible_items, dataset_path)


def load_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".mat":
        return load_mat_dataset(dataset_path)
    if suffix == ".csv":
        return load_csv_dataset(dataset_path)
    if suffix == ".npz":
        return load_npz_dataset(dataset_path)
    raise ValueError(f"Unsupported dataset format: {suffix}")
