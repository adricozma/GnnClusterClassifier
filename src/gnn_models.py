"""Graph Neural Network architectures with local phase-space relational encoding for Study 2.

Implements:
- `compute_node_relational_features`: On-the-fly O(E) extraction of ego-relative astrometric
  residuals, local k-NN neighborhood dispersion, mean neighbor distance, and the Casertano & Hut
  (1985) local phase-space log-density estimator directly from `(data.x, data.edge_index)`.
- `GCN`: Three-layer Residual Graph Convolutional Network with LayerNorm and ego-relational readout.
- `GAT`: Three-layer Multi-Head Residual Graph Attention Network (`heads=4`) with LayerNorm and
  ego-relational readout.
"""

from __future__ import annotations

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

    def __init__(self, in_channels: int = 5, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = float(dropout)
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
        in_rel = 3 * x.shape[1] + 3
        h0 = getattr(data, "h0", None)
        if h0 is None or h0.shape[0] != x.shape[0] or h0.shape[1] != in_rel:
            h0 = compute_node_relational_features(x, edge_index)

        h1 = F.gelu(self.norm1(self.conv1(h0, edge_index) + self.skip1(h0)))
        if self.dropout > 0.0:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.gelu(self.norm2(self.conv2(h1, edge_index) + self.skip2(h1)))
        if self.dropout > 0.0:
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
        out_graph = self.conv3(h2, edge_index)
        ego_in = torch.cat([h2, h0], dim=-1)
        if self.dropout > 0.0:
            ego_in = F.dropout(ego_in, p=self.dropout, training=self.training)
        out_ego = self.ego_head(ego_in)
        return out_graph + out_ego


class GAT(nn.Module):
    """Three-layer Multi-Head Residual Graph Attention Network with local phase-space relational features."""

    def __init__(self, in_channels: int = 5, dropout: float = 0.2, edge_dropout: float | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = float(dropout)
        self.edge_dropout = float(min(self.dropout, 0.15) if edge_dropout is None else edge_dropout)
        in_rel = 3 * in_channels + 3

        self.conv1 = GATConv(in_rel, 32, heads=4, concat=True, dropout=self.edge_dropout)
        self.skip1 = nn.Linear(in_rel, 128)
        self.norm1 = nn.LayerNorm(128)

        self.conv2 = GATConv(128, 64, heads=4, concat=True, dropout=self.edge_dropout)
        self.skip2 = nn.Linear(128, 256)
        self.norm2 = nn.LayerNorm(256)

        # Final 1-head scalar logit readout layer: keep edge_dropout=0.0 to avoid randomly zeroing final logits
        self.conv3 = GATConv(256, 1, heads=1, concat=False, dropout=0.0)
        self.ego_head = nn.Sequential(
            nn.Linear(256 + in_rel, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        in_rel = 3 * x.shape[1] + 3
        h0 = getattr(data, "h0", None)
        if h0 is None or h0.shape[0] != x.shape[0] or h0.shape[1] != in_rel:
            h0 = compute_node_relational_features(x, edge_index)

        h1 = F.elu(self.norm1(self.conv1(h0, edge_index) + self.skip1(h0)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.elu(self.norm2(self.conv2(h1, edge_index) + self.skip2(h1)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        out_graph = self.conv3(h2, edge_index)
        ego_in = F.dropout(torch.cat([h2, h0], dim=-1), p=self.dropout, training=self.training)
        out_ego = self.ego_head(ego_in)
        return out_graph + out_ego


def load_pretrained_with_channel_adaptation(
    model: nn.Module,
    state_dict: dict,
    selected_features: list[int] | None = None,
) -> None:
    """Load pre-trained weights into `model`, adapting relational input channels (e.g., 5D -> 4D) if needed.

    In `compute_node_relational_features(x, edge_index)` with `d_src` input features, the output `h0`
    has dimension `3 * d_src + 3` ordered as:
      - `[0 : d_src]`             : raw features `x`
      - `[d_src : 2*d_src]`       : ego-relative residual `dx_ego`
      - `[2*d_src : 3*d_src]`     : local neighborhood std `neigh_std`
      - `[3*d_src : 3*d_src + 3]` : `mean_dist`, `log_density`, `radial_norm`

    When transferring from a `d_src`-channel checkpoint (e.g. 5 observables, `in_rel=18`) to a
    `d_dst`-channel model (e.g. 4 observables, `in_rel=15`), the exact corresponding 15 columns
    of `conv1.lin.weight`, `skip1.weight`, and the trailing `h0` slice of `ego_head.0.weight`
    are extracted and scaled by `sqrt(in_rel_src / in_rel_dst)` to preserve activation variance.
    """
    clean_sd = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        clean_sd[clean_k] = v

    model_sd = model.state_dict()
    dst_in_rel = model_sd["skip1.weight"].shape[1]
    src_in_rel = clean_sd["skip1.weight"].shape[1]

    if src_in_rel == dst_in_rel:
        model.load_state_dict(clean_sd)
        return

    d_src = (src_in_rel - 3) // 3
    d_dst = (dst_in_rel - 3) // 3

    if selected_features is not None and len(selected_features) == d_dst:
        feat_idx = list(selected_features)
    else:
        feat_idx = list(range(d_dst))

    rel_indices = (
        [i for i in feat_idx]
        + [d_src + i for i in feat_idx]
        + [2 * d_src + i for i in feat_idx]
        + [3 * d_src, 3 * d_src + 1, 3 * d_src + 2]
    )
    idx_tensor = torch.tensor(rel_indices, dtype=torch.long)
    scale = (float(src_in_rel) / float(dst_in_rel)) ** 0.5

    adapted_sd = {}
    for k, target_param in model_sd.items():
        if k not in clean_sd:
            adapted_sd[k] = target_param
            continue
        src_param = clean_sd[k]
        if src_param.shape == target_param.shape:
            adapted_sd[k] = src_param
        elif k in ("conv1.lin.weight", "skip1.weight") and src_param.shape[1] == src_in_rel:
            adapted_sd[k] = src_param[:, idx_tensor.to(src_param.device)] * scale
        elif k == "ego_head.0.weight" and src_param.shape[1] - src_in_rel == target_param.shape[1] - dst_in_rel:
            h2_dim = target_param.shape[1] - dst_in_rel
            h2_part = src_param[:, :h2_dim]
            h0_part = src_param[:, h2_dim:][:, idx_tensor.to(src_param.device)] * scale
            adapted_sd[k] = torch.cat([h2_part, h0_part], dim=1)
        else:
            raise RuntimeError(
                f"Cannot adapt parameter '{k}' from shape {tuple(src_param.shape)} to {tuple(target_param.shape)}"
            )

    model.load_state_dict(adapted_sd)

