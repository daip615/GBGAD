from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float = 0.0,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.encoder = build_mlp(input_dim, hidden_dims, latent_dim, dropout=dropout)
        self.decoder = build_mlp(latent_dim, list(reversed(hidden_dims)), input_dim, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1)
        inv_sqrt_degree = torch.pow(degree.clamp_min(1e-8), -0.5)
        norm_adj = inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)
        propagated = norm_adj @ features
        return self.activation(self.linear(self.dropout(propagated)))


@dataclass
class GBGADOutput:
    latent: torch.Tensor
    reconstruction: torch.Tensor
    ball_embeddings: torch.Tensor
    sample_scores: torch.Tensor
    score_logits: torch.Tensor


class GBGAD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        ae_hidden_dims: list[int] | None = None,
        gnn_hidden_dim: int = 32,
        gnn_layers: int = 2,
        scorer_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ae_hidden_dims = ae_hidden_dims or [128, 64]
        scorer_hidden_dims = scorer_hidden_dims or [64, 32]

        self.autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=ae_hidden_dims, dropout=dropout)

        convs: list[nn.Module] = []
        in_dim = latent_dim + 5
        for _ in range(gnn_layers):
            convs.append(GraphConv(in_dim, gnn_hidden_dim, dropout))
            in_dim = gnn_hidden_dim
        self.gnn_layers = nn.ModuleList(convs)

        # The final GBGAD scorer fuses the sample latent representation z_i
        # and the assigned granular-ball graph representation h_b(i).  The
        # normalized within-ball deviation is retained by the trainer only for
        # constructing the teacher signal; it is not part of the scoring vector.
        scorer_input_dim = latent_dim + gnn_hidden_dim
        self.scorer = build_mlp(scorer_input_dim, scorer_hidden_dims, 1, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        ball_node_features: torch.Tensor,
        adjacency: torch.Tensor,
        ball_assignments: torch.Tensor,
        normalized_offsets: torch.Tensor,
    ) -> GBGADOutput:
        latent, reconstruction = self.autoencoder(x)

        h = ball_node_features
        for layer in self.gnn_layers:
            h = layer(h, adjacency)

        sample_ball_embeddings = h[ball_assignments]
        sample_features = torch.cat([latent, sample_ball_embeddings], dim=1)
        logits = self.scorer(sample_features).squeeze(1)
        scores = torch.sigmoid(logits)

        return GBGADOutput(
            latent=latent,
            reconstruction=reconstruction,
            ball_embeddings=h,
            sample_scores=scores,
            score_logits=logits,
        )


# Backward-compatible aliases keep the copied trainer module focused on the
# controlled scoring-vector change while exposing GBGAD as the final model name.
GBGNNOutput = GBGADOutput
GBGNN = GBGAD
