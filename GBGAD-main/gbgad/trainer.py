from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim

from .balls import GranularBall, build_granular_balls
from .losses import compactness_loss, margin_loss, pseudo_label_loss, reconstruction_loss
from .model import GBGAD


@dataclass
class GBGADConfig:
    latent_dim: int = 8
    ae_hidden_dims: tuple[int, ...] = (128, 64)
    gnn_hidden_dim: int = 32
    gnn_layers: int = 2
    scorer_hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    pretrain_epochs: int = 20
    ball_update_interval: int = 5
    graph_k: int = 5
    min_ball_size: int = 5
    recon_weight: float = 1.0
    compact_weight: float = 0.2
    pseudo_weight: float = 0.5
    # The final GBGAD objective does not use the ranking-margin loss.
    # This option is retained only so that earlier controlled ablations can
    # still be reproduced explicitly by setting a positive value.
    margin_weight: float = 0.0
    pseudo_ratio: float = 0.15
    teacher_rec_weight: float = 0.33
    offset_epsilon: float = 1e-6
    margin: float = 0.2
    device: str = "cpu"
    random_state: int = 42


@dataclass
class TrainResult:
    scores: np.ndarray
    latent: np.ndarray
    ball_assignments: np.ndarray
    balls: list[GranularBall]
    history: list[dict[str, float]]


class GBGADTrainer:
    def __init__(self, config: GBGADConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model: GBGAD | None = None
        self.device = torch.device(config.device)
        torch.manual_seed(config.random_state)
        np.random.seed(config.random_state)

    def _sample_reconstruction_errors(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        return torch.norm(inputs - reconstructions, dim=1).pow(2)

    def _minmax_normalize(self, values: torch.Tensor) -> torch.Tensor:
        min_value = values.min()
        max_value = values.max()
        return (values - min_value) / (max_value - min_value + 1e-8)

    def _teacher_scores(
        self,
        reconstruction_errors: torch.Tensor,
        normalized_offsets: torch.Tensor,
        sample_isolation: torch.Tensor,
    ) -> torch.Tensor:
        rec_term = self._minmax_normalize(reconstruction_errors)
        offset_term = self._minmax_normalize(normalized_offsets)
        isolation_term = self._minmax_normalize(sample_isolation)
        alpha = self.config.teacher_rec_weight
        return alpha * rec_term + (1.0 - alpha) * 0.5 * (offset_term + isolation_term)

    def _pseudo_masks(self, teacher_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_q = torch.quantile(teacher_scores, self.config.pseudo_ratio)
        high_q = torch.quantile(teacher_scores, 1.0 - self.config.pseudo_ratio)
        normal_mask = teacher_scores <= low_q
        anomaly_mask = teacher_scores >= high_q
        return normal_mask, anomaly_mask

    def _build_graph_inputs(
        self,
        latent: torch.Tensor,
        recon_errors: torch.Tensor,
        balls: list[GranularBall],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = torch.stack([ball.center for ball in balls], dim=0)
        radii = torch.tensor([ball.radius for ball in balls], device=latent.device, dtype=latent.dtype)
        sizes = torch.tensor([len(ball.sample_indices) for ball in balls], device=latent.device, dtype=latent.dtype)
        dispersions = torch.tensor([ball.compactness for ball in balls], device=latent.device, dtype=latent.dtype)

        ball_count = len(balls)
        graph_distance = torch.zeros(ball_count, ball_count, device=latent.device, dtype=latent.dtype)
        adjacency = torch.eye(ball_count, device=latent.device, dtype=latent.dtype)
        isolation = torch.zeros(ball_count, device=latent.device, dtype=latent.dtype)
        ball_avg_recon = torch.zeros(ball_count, device=latent.device, dtype=latent.dtype)

        for ball in balls:
            ball_avg_recon[ball.index] = recon_errors[ball.sample_indices].mean()

        if ball_count > 1:
            # d_mn = ||c_m - c_n||_2: graph distances depend only on centers.
            graph_distance = torch.cdist(centers, centers, p=2)
            neighbor_count = min(self.config.graph_k, ball_count - 1)
            positive_distances = graph_distance[graph_distance > 0]
            sigma = positive_distances.mean().item() if positive_distances.numel() > 0 else 1.0
            sigma = max(sigma, 1e-6)
            neighbor_distances = graph_distance.clone()
            neighbor_distances.fill_diagonal_(float("inf"))

            for idx in range(ball_count):
                nearest = torch.argsort(neighbor_distances[idx])[:neighbor_count]
                isolation[idx] = graph_distance[idx, nearest].mean()
                weights = torch.exp(-(graph_distance[idx, nearest].pow(2)) / (2 * sigma * sigma))
                adjacency[idx, nearest] = torch.maximum(adjacency[idx, nearest], weights)
                adjacency[nearest, idx] = torch.maximum(adjacency[nearest, idx], weights)

        node_features = torch.cat(
            [
                centers,
                radii.unsqueeze(1),
                torch.log1p(sizes).unsqueeze(1),
                dispersions.unsqueeze(1),
                ball_avg_recon.unsqueeze(1),
                isolation.unsqueeze(1),
            ],
            dim=1,
        )

        sample_to_ball = torch.empty(latent.size(0), dtype=torch.long, device=latent.device)
        normalized_offsets = torch.empty(latent.size(0), dtype=latent.dtype, device=latent.device)
        sample_isolation = torch.empty(latent.size(0), dtype=latent.dtype, device=latent.device)
        for ball in balls:
            sample_to_ball[ball.sample_indices] = ball.index
            offset = torch.norm(latent[ball.sample_indices] - centers[ball.index], dim=1)
            normalized_offsets[ball.sample_indices] = offset / (ball.radius + self.config.offset_epsilon)
            sample_isolation[ball.sample_indices] = isolation[ball.index]

        return node_features, adjacency, sample_to_ball, normalized_offsets, centers, sample_isolation

    def _pretrain_autoencoder(self, x: torch.Tensor) -> None:
        assert self.model is not None
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        for _ in range(self.config.pretrain_epochs):
            self.model.train()
            optimizer.zero_grad()
            _, reconstruction = self.model.autoencoder(x)
            loss = reconstruction_loss(x, reconstruction)
            loss.backward()
            optimizer.step()

    def fit(self, x: np.ndarray) -> TrainResult:
        x_scaled = self.scaler.fit_transform(x).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled).to(self.device)

        self.model = GBGAD(
            input_dim=x_tensor.size(1),
            latent_dim=self.config.latent_dim,
            ae_hidden_dims=list(self.config.ae_hidden_dims),
            gnn_hidden_dim=self.config.gnn_hidden_dim,
            gnn_layers=self.config.gnn_layers,
            scorer_hidden_dims=list(self.config.scorer_hidden_dims),
            dropout=self.config.dropout,
        ).to(self.device)

        self._pretrain_autoencoder(x_tensor)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        history: list[dict[str, float]] = []
        balls: list[GranularBall] = []
        node_features = adjacency = assignments = offsets = ball_centers = sample_isolation = None

        for epoch in range(self.config.epochs):
            assert self.model is not None
            self.model.train()

            refresh_structure = epoch % self.config.ball_update_interval == 0 or node_features is None
            if refresh_structure:
                with torch.no_grad():
                    latent_snapshot, recon_snapshot = self.model.autoencoder(x_tensor)
                    recon_errors_snapshot = self._sample_reconstruction_errors(x_tensor, recon_snapshot)
                balls = build_granular_balls(latent_snapshot.detach(), min_ball_size=self.config.min_ball_size)
                node_features, adjacency, assignments, offsets, ball_centers, sample_isolation = self._build_graph_inputs(
                    latent_snapshot.detach(), recon_errors_snapshot.detach(), balls
                )

            optimizer.zero_grad()
            output = self.model(
                x=x_tensor,
                ball_node_features=node_features,
                adjacency=adjacency,
                ball_assignments=assignments,
                normalized_offsets=offsets,
            )

            recon_errors = self._sample_reconstruction_errors(x_tensor, output.reconstruction)
            teacher_scores = self._teacher_scores(recon_errors.detach(), offsets.detach(), sample_isolation.detach())
            normal_mask, anomaly_mask = self._pseudo_masks(teacher_scores)

            recon = reconstruction_loss(x_tensor, output.reconstruction)
            compact = compactness_loss(output.latent, assignments, ball_centers, normal_mask)
            pseudo = pseudo_label_loss(output.score_logits, normal_mask, anomaly_mask)
            loss = (
                self.config.recon_weight * recon
                + self.config.compact_weight * compact
                + self.config.pseudo_weight * pseudo
            )
            boundary = output.sample_scores.new_tensor(0.0)
            if self.config.margin_weight > 0.0:
                boundary = margin_loss(
                    output.sample_scores,
                    normal_mask,
                    anomaly_mask,
                    margin=self.config.margin,
                )
                loss = loss + self.config.margin_weight * boundary
            loss.backward()
            optimizer.step()

            epoch_record = {
                "epoch": float(epoch + 1),
                "loss": float(loss.item()),
                "recon": float(recon.item()),
                "compact": float(compact.item()),
                "pseudo": float(pseudo.item()),
                "balls": float(len(balls)),
            }
            if self.config.margin_weight > 0.0:
                epoch_record["margin"] = float(boundary.item())
            history.append(epoch_record)

        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            latent, recon = self.model.autoencoder(x_tensor)
            recon_errors = self._sample_reconstruction_errors(x_tensor, recon)
            balls = build_granular_balls(latent.detach(), min_ball_size=self.config.min_ball_size)
            node_features, adjacency, assignments, offsets, _, _ = self._build_graph_inputs(latent.detach(), recon_errors.detach(), balls)
            output = self.model(
                x=x_tensor,
                ball_node_features=node_features,
                adjacency=adjacency,
                ball_assignments=assignments,
                normalized_offsets=offsets,
            )

        return TrainResult(
            scores=output.sample_scores.cpu().numpy(),
            latent=output.latent.cpu().numpy(),
            ball_assignments=assignments.cpu().numpy(),
            balls=balls,
            history=history,
        )

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).scores


# Compatibility aliases for code that previously imported the copied GBGNN
# class names from this package.
GBGNNConfig = GBGADConfig
GBGNNTrainer = GBGADTrainer
