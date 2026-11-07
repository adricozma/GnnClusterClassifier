"""Graph Neural Network architectures with local phase-space relational encoding for Study 2.

Implements:
- `compute_node_relational_features`: On-the-fly O(E) extraction of ego-relative astrometric
  residuals, local k-NN neighborhood dispersion, mean neighbor distance, and the Casertano & Hut
  (1985) local phase-space log-density estimator directly from `(data.x, data.edge_index)`.
- `GCN`: Three-layer Residual Graph Convolutional Network with LayerNorm and ego-relational readout.
- `GAT`: Three-layer Multi-Head Residual Graph Attention Network (`heads=4`) with LayerNorm and
  ego-relational readout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


def compute_node_relational_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Augment node features x (N, d) with local k-NN phase-space relational invariants.

    For a k-NN graph with constant topological degree k, standard message passing is blind to
    the metric scale of edges ||x_j - x_i||_2. This function computes exact per-node local
    neighborhood statistics:
      1. Raw standardized observables: x_i (d)
      2. Ego-relative kinematic/spatial residual: x_i - <x>_{N(i)} (d)
      3. Local neighborhood standard deviation: sigma_{N(i)} (d)
      4. Mean k-NN edge length: <||x_j - x_i||_2>_{N(i)} (1)
      5. Casertano & Hut (1985) phase-space log-density proxy: -d * ln(<d_ij> + eps) / 10 (1)
      6. Global standardized phase-space radius from stream center: ||x_i||_2 (1)

    Total output dimension: `3 * d + 3`.
    """
    n_nodes, d = x.shape
    if edge_index.numel() == 0:
        zeros_d = torch.zeros_like(x)
        zeros_1 = torch.zeros((n_nodes, 1), dtype=x.dtype, device=x.device)
        radial_norm = torch.norm(x, dim=-1, keepdim=True)
        return torch.cat([x, zeros_d, zeros_d, zeros_1, zeros_1, radial_norm], dim=-1)

    src, dst = edge_index[0], edge_index[1]
    ones = torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device)
    deg = torch.zeros((n_nodes, 1), dtype=x.dtype, device=x.device)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1.0)

    neigh_sum = torch.zeros_like(x)
    neigh_sum.index_add_(0, dst, x[src])
    neigh_mean = neigh_sum / deg

    dx_ego = x - neigh_mean

    diff_sq = (x[src] - neigh_mean[dst]) ** 2
    neigh_var = torch.zeros_like(x)
    neigh_var.index_add_(0, dst, diff_sq)
    neigh_std = torch.sqrt(neigh_var / deg + 1e-8)

    edge_dist = torch.norm(x[src] - x[dst], dim=-1, keepdim=True)
    dist_sum = torch.zeros((n_nodes, 1), dtype=x.dtype, device=x.device)
    dist_sum.index_add_(0, dst, edge_dist)
    mean_dist = dist_sum / deg
    log_density = (-float(d) * torch.log(mean_dist + 1e-4)) * 0.1

    radial_norm = torch.norm(x, dim=-1, keepdim=True)

    return torch.cat([x, dx_ego, neigh_std, mean_dist, log_density, radial_norm], dim=-1)


class GCN(nn.Module):
    """Three-layer Residual Graph Convolutional Network with local phase-space relational features."""

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.in_channels = in_channels
        in_rel = 3 * in_channels + 3

        self.conv1 = GCNConv(in_rel, 64)
        self.skip1 = nn.Linear(in_rel, 64)
        self.norm1 = nn.LayerNorm(64)

        self.conv2 = GCNConv(64, 128)
        self.skip2 = nn.Linear(64, 128)
        self.norm2 = nn.LayerNorm(128)

        self.conv3 = GCNConv(128, 1)
        self.ego_head = nn.Sequential(
            nn.Linear(128 + in_rel, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h0 = compute_node_relational_features(x, edge_index)

        h1 = F.gelu(self.norm1(self.conv1(h0, edge_index) + self.skip1(h0)))
        h2 = F.gelu(self.norm2(self.conv2(h1, edge_index) + self.skip2(h1)))
        out_graph = self.conv3(h2, edge_index)
        out_ego = self.ego_head(torch.cat([h2, h0], dim=-1))
        return out_graph + out_ego


class GAT(nn.Module):
    """Three-layer Multi-Head Residual Graph Attention Network with local phase-space relational features."""

    def __init__(self, in_channels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        in_rel = 3 * in_channels + 3

        self.conv1 = GATConv(in_rel, 32, heads=4, concat=True, dropout=dropout)
        self.skip1 = nn.Linear(in_rel, 128)
        self.norm1 = nn.LayerNorm(128)

        self.conv2 = GATConv(128, 64, heads=4, concat=True, dropout=dropout)
        self.skip2 = nn.Linear(128, 256)
        self.norm2 = nn.LayerNorm(256)

        self.conv3 = GATConv(256, 1, heads=1, concat=False, dropout=dropout)
        self.ego_head = nn.Sequential(
            nn.Linear(256 + in_rel, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h0 = compute_node_relational_features(x, edge_index)

        h1 = F.elu(self.norm1(self.conv1(h0, edge_index) + self.skip1(h0)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.elu(self.norm2(self.conv2(h1, edge_index) + self.skip2(h1)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        out_graph = self.conv3(h2, edge_index)
        out_ego = self.ego_head(torch.cat([h2, h0], dim=-1))
        return out_graph + out_ego
