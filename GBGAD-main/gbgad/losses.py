from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstructions, inputs)


def compactness_loss(
    latent: torch.Tensor,
    ball_assignments: torch.Tensor,
    ball_centers: torch.Tensor,
    normal_mask: torch.Tensor,
) -> torch.Tensor:
    if normal_mask.sum() == 0:
        return latent.new_tensor(0.0)
    distances = torch.norm(latent - ball_centers[ball_assignments], dim=1).pow(2)
    return distances[normal_mask].mean()


def pseudo_label_loss(
    logits: torch.Tensor,
    normal_mask: torch.Tensor,
    anomaly_mask: torch.Tensor,
) -> torch.Tensor:
    selected_mask = normal_mask | anomaly_mask
    if selected_mask.sum() == 0:
        return logits.new_tensor(0.0)
    targets = anomaly_mask[selected_mask].float()
    return F.binary_cross_entropy_with_logits(logits[selected_mask], targets)


def margin_loss(
    scores: torch.Tensor,
    normal_mask: torch.Tensor,
    anomaly_mask: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Legacy ranking loss retained only for controlled ablation runs."""
    if normal_mask.sum() == 0 or anomaly_mask.sum() == 0:
        return scores.new_tensor(0.0)

    normal_scores = torch.sort(scores[normal_mask])[0]
    anomaly_scores = torch.sort(scores[anomaly_mask], descending=True)[0]
    pair_count = min(len(normal_scores), len(anomaly_scores))
    if pair_count == 0:
        return scores.new_tensor(0.0)
    return F.relu(margin - (anomaly_scores[:pair_count] - normal_scores[:pair_count])).mean()
