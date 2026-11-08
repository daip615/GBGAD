from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_data import load_experiment_dataset
from gbgad import GBGADConfig, GBGADTrainer


SUPPORTED_SUFFIXES = {".csv", ".mat", ".npz"}
METHOD_NAME = "GBGAD"


def discover_datasets(data_path: Path) -> list[Path]:
    if data_path.is_file():
        if data_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported dataset file: {data_path}")
        return [data_path]
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    return sorted(
        path
        for path in data_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def normalize_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("Labels are empty.")
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size != 2:
        raise ValueError("Evaluation requires binary labels.")
    normal_label = unique[int(np.argmax(counts))]
    normalized = (labels != normal_label).astype(np.int64)
    if normalized.sum() == 0 or normalized.sum() == normalized.size:
        raise ValueError("Failed to infer normal and anomalous labels.")
    return normalized


def load_labels(
    dataset_labels: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    if dataset_labels is None:
        return None, ""
    return normalize_binary_labels(dataset_labels), "loader"


def build_config(args: argparse.Namespace) -> GBGADConfig:
    return GBGADConfig(
        latent_dim=args.latent_dim,
        ae_hidden_dims=tuple(args.ae_hidden_dims),
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        scorer_hidden_dims=tuple(args.scorer_hidden_dims),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        ball_update_interval=args.ball_update_interval,
        graph_k=args.graph_k,
        min_ball_size=args.min_ball_size,
        recon_weight=args.recon_weight,
        compact_weight=args.compact_weight,
        pseudo_weight=args.pseudo_weight,
        pseudo_ratio=args.pseudo_ratio,
        teacher_rec_weight=args.teacher_rec_weight,
        offset_epsilon=args.offset_epsilon,
        device=args.device,
        random_state=args.seed,
    )


def save_scores(
    output_path: Path,
    scores: np.ndarray,
    ball_assignments: np.ndarray,
    labels: np.ndarray | None,
) -> None:
    columns = [
        np.arange(scores.shape[0], dtype=np.int64),
        scores,
        ball_assignments,
    ]
    header = "index,score,ball_id"
    formats: list[str] = ["%d", "%.8f", "%d"]
    if labels is not None:
        columns.append(labels.astype(np.int64))
        header += ",label"
        formats.append("%d")
    np.savetxt(
        output_path,
        np.column_stack(columns),
        delimiter=",",
        fmt=formats,
        header=header,
        comments="",
    )


def evaluate_if_possible(
    labels: np.ndarray | None,
    scores: np.ndarray,
) -> tuple[float | None, float | None]:
    if labels is None or np.unique(labels).size <= 1:
        return None, None
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "scoring_vector",
        "training_protocol",
        "label_exclusion",
        "dataset",
        "status",
        "samples",
        "features",
        "M",
        "M_over_N",
        "seed",
        "roc_auc",
        "average_precision",
        "final_loss",
        "feature_key",
        "label_key",
        "score_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export anomaly scores for the final GBGAD model. The training "
            "objective contains reconstruction, granular-ball compactness, "
            "and pseudo-label binary cross-entropy losses."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("datasets") / "Classical")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "GBGAD",
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--ae-hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--scorer-hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--ball-update-interval", type=int, default=5)
    parser.add_argument("--graph-k", type=int, default=5)
    parser.add_argument("--min-ball-size", type=int, default=5)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--compact-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-ratio", type=float, default=0.15)
    parser.add_argument("--teacher-rec-weight", type=float, default=0.33)
    parser.add_argument("--offset-epsilon", type=float, default=1e-6)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.pretrain_epochs < 0:
        raise ValueError("--pretrain-epochs must be non-negative.")
    if args.ball_update_interval < 1:
        raise ValueError("--ball-update-interval must be at least 1.")
    if args.graph_k < 1:
        raise ValueError("--graph-k must be at least 1.")
    if args.min_ball_size < 2:
        raise ValueError("--min-ball-size must be at least 2.")
    if args.gnn_layers < 1:
        raise ValueError("--gnn-layers must be at least 1.")
    if not 0.0 < args.pseudo_ratio < 0.5:
        raise ValueError("--pseudo-ratio must lie in (0, 0.5).")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_paths = discover_datasets(args.data)
    if not dataset_paths:
        raise ValueError(f"No supported datasets found in {args.data}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output_dir / "gbgad_summary.csv"
    rows: list[dict[str, object]] = []

    for dataset_path in dataset_paths:
        row: dict[str, object] = {
            "method": METHOD_NAME,
            "scoring_vector": "[z_i,h_b(i)]",
            "training_protocol": "full_unlabeled_transductive",
            "label_exclusion": "excluded_from_features_and_training_targets",
            "dataset": dataset_path.name,
            "seed": args.seed,
        }
        try:
            dataset = load_experiment_dataset(dataset_path)
            labels, label_source = load_labels(dataset.labels)
            if dataset.label_key:
                label_source = dataset.label_key

            config = build_config(args)
            trainer = GBGADTrainer(config)
            result = trainer.fit(dataset.features)
            scores = np.asarray(result.scores, dtype=np.float64).reshape(-1)
            assignments = np.asarray(result.ball_assignments, dtype=np.int64).reshape(-1)
            sample_count = int(dataset.features.shape[0])
            ball_count = int(len(result.balls))

            if scores.shape[0] != sample_count:
                raise ValueError(f"{METHOD_NAME} returned {scores.shape[0]} scores for {sample_count} samples.")
            if not np.all(np.isfinite(scores)):
                raise ValueError(f"{METHOD_NAME} produced non-finite anomaly scores.")
            if np.any(scores < 0.0) or np.any(scores > 1.0):
                raise ValueError(f"{METHOD_NAME} anomaly scores must lie in [0, 1].")
            if ball_count < 1:
                raise ValueError(f"{METHOD_NAME} did not produce any granular balls.")
            if assignments.shape[0] != sample_count:
                raise ValueError("The number of ball assignments does not match the sample count.")
            if assignments.min() < 0 or assignments.max() >= ball_count:
                raise ValueError("A ball assignment lies outside [0, M-1].")

            output_path = args.output_dir / f"{dataset_path.stem}.csv"
            save_scores(output_path, scores, assignments, labels)
            roc_auc, average_precision = evaluate_if_possible(labels, scores)
            final_loss = result.history[-1]["loss"] if result.history else float("nan")
            row.update(
                {
                    "status": "ok",
                    "samples": sample_count,
                    "features": int(dataset.features.shape[1]),
                    "M": ball_count,
                    "M_over_N": f"{ball_count / sample_count:.8f}",
                    "roc_auc": "" if roc_auc is None else f"{roc_auc:.8f}",
                    "average_precision": "" if average_precision is None else f"{average_precision:.8f}",
                    "final_loss": f"{final_loss:.8f}",
                    "feature_key": dataset.feature_key,
                    "label_key": label_source,
                    "score_path": str(output_path),
                    "error": "",
                }
            )
            metric_text = ""
            if roc_auc is not None:
                metric_text = (
                    f" | roc_auc={roc_auc:.6f}"
                    f" | average_precision={average_precision:.6f}"
                )
            print(
                f"Saved {output_path} | method={METHOD_NAME} | N={sample_count} | M={ball_count}"
                f" | M/N={ball_count / sample_count:.6f}" + metric_text
            )
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            write_summary(summary_path, rows)
            print(f"Failed {dataset_path}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        else:
            rows.append(row)
            write_summary(summary_path, rows)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
