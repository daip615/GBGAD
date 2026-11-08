from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GranularBall:
    index: int
    sample_indices: torch.Tensor
    center: torch.Tensor
    radius: float
    sum_distance: float
    compactness: float


def _ball_statistics(points: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
    center = points.mean(dim=0)
    distances = torch.norm(points - center, dim=1)
    radius = distances.mean().item() if len(points) > 0 else 0.0
    sum_distance = distances.sum().item()
    compactness = distances.pow(2).mean().item() if len(points) > 0 else 0.0
    return center, radius, sum_distance, compactness


def _initialize_two_means(
    points: torch.Tensor,
    max_pairwise_elements: int = 64 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the global farthest pair without storing an oversized NxN matrix."""
    sample_count = points.size(0)
    row_chunk_size = max(
        1,
        min(sample_count, max_pairwise_elements // max(sample_count, 1)),
    )

    best_distance = points.new_tensor(-1.0)
    seed_a = torch.tensor(0, device=points.device, dtype=torch.long)
    seed_b = torch.tensor(0, device=points.device, dtype=torch.long)
    for start in range(0, sample_count, row_chunk_size):
        stop = min(start + row_chunk_size, sample_count)
        distances = torch.cdist(points[start:stop], points, p=2)
        local_flat_index = torch.argmax(distances)
        local_distance = distances.reshape(-1)[local_flat_index]
        if local_distance > best_distance:
            local_row = local_flat_index // sample_count
            local_column = local_flat_index % sample_count
            best_distance = local_distance
            seed_a = local_row + start
            seed_b = local_column
    return points[seed_a].clone(), points[seed_b].clone()


def _two_means(points: torch.Tensor, iterations: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    if points.size(0) <= 2:
        midpoint = max(1, points.size(0) // 2)
        left = torch.arange(midpoint, device=points.device)
        right = torch.arange(midpoint, points.size(0), device=points.device)
        return left, right

    center_a, center_b = _initialize_two_means(points)
    assignments = None
    for _ in range(iterations):
        dist_a = torch.norm(points - center_a, dim=1)
        dist_b = torch.norm(points - center_b, dim=1)
        current = dist_a <= dist_b
        if current.all() or (~current).all():
            order = torch.argsort(dist_a)
            midpoint = max(1, len(order) // 2)
            current = torch.zeros(points.size(0), dtype=torch.bool, device=points.device)
            current[order[:midpoint]] = True
        if assignments is not None and torch.equal(current, assignments):
            break
        assignments = current
        center_a = points[assignments].mean(dim=0)
        center_b = points[~assignments].mean(dim=0)

    left = torch.nonzero(assignments, as_tuple=False).flatten()
    right = torch.nonzero(~assignments, as_tuple=False).flatten()
    return left, right


def _split_if_valid(points: torch.Tensor, min_ball_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if points.size(0) < 2 * min_ball_size:
        return None
    left, right = _two_means(points)
    if len(left) < min_ball_size or len(right) < min_ball_size:
        return None
    return left, right


def _sd_prefers_split(points: torch.Tensor, min_ball_size: int) -> tuple[bool, tuple[torch.Tensor, torch.Tensor] | None]:
    split = _split_if_valid(points, min_ball_size)
    if split is None:
        return False, None
    left, right = split
    _, _, parent_sd, _ = _ball_statistics(points)
    _, _, left_sd, _ = _ball_statistics(points[left])
    _, _, right_sd, _ = _ball_statistics(points[right])
    return (left_sd + right_sd) < parent_sd, split


def _force_large_radius_split(
    leaves: list[torch.Tensor],
    embeddings: torch.Tensor,
    min_ball_size: int,
) -> list[torch.Tensor]:
    if not leaves:
        return leaves

    updated = leaves[:]
    changed = True
    while changed:
        changed = False
        radii = []
        for indices in updated:
            _, radius, _, _ = _ball_statistics(embeddings[indices])
            radii.append(radius)
        radius_tensor = torch.tensor(radii, device=embeddings.device, dtype=embeddings.dtype)
        radius_threshold = 2.0 * max(radius_tensor.mean().item(), radius_tensor.median().item())

        next_leaves: list[torch.Tensor] = []
        for indices, radius in zip(updated, radii):
            points = embeddings[indices]
            if radius > radius_threshold and points.size(0) >= 2 * min_ball_size:
                split = _split_if_valid(points, min_ball_size)
                if split is not None:
                    left, right = split
                    next_leaves.append(indices[left])
                    next_leaves.append(indices[right])
                    changed = True
                    continue
            next_leaves.append(indices)
        updated = next_leaves
    return updated


def build_granular_balls(
    embeddings: torch.Tensor,
    min_ball_size: int = 5,
) -> list[GranularBall]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if embeddings.size(0) == 0:
        return []

    pending = [torch.arange(embeddings.size(0), device=embeddings.device)]
    leaves: list[torch.Tensor] = []

    while pending:
        current_indices = pending.pop()
        points = embeddings[current_indices]
        should_split, split = _sd_prefers_split(points, min_ball_size)
        if should_split and split is not None:
            left, right = split
            pending.append(current_indices[left])
            pending.append(current_indices[right])
        else:
            leaves.append(current_indices)

    leaves = _force_large_radius_split(leaves, embeddings, min_ball_size)

    balls: list[GranularBall] = []
    for idx, sample_indices in enumerate(leaves):
        center, radius, sum_distance, compactness = _ball_statistics(embeddings[sample_indices])
        balls.append(
            GranularBall(
                index=idx,
                sample_indices=sample_indices,
                center=center,
                radius=radius,
                sum_distance=sum_distance,
                compactness=compactness,
            )
        )
    return balls
