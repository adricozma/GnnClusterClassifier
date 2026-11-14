"""V2 Graph Attention Network architectures and N-invariant relational builders.

Provides Study 2 V2 models strictly isolated from V1 (`src/gnn_models.py`):
- Pure-PyTorch neighbor aggregation (`index_add_`) with N-invariant local sky
  surface density normalization so completeness cuts (G <= 20.7) and random
  subsampling leave density features invariant.
- Support for 2D sky edge attributes `[dRA*, dDec, theta_sky]` injected directly
  into `GATConv(..., edge_dim=3)`.
- Support for 4 per-star astrometric uncertainty / parallax-SNR channels (`err_x`).
- 3-class node logits (`0: Unbound/Field`, `1: Potential Escaper`, `2: Bound Core`)
  or binary logits (`out_channels=1`).
- Variance-preserving `5D -> 4D` subspace weight transfer aligned with the
  `(x, x - mean, std, sky_scalars, err_x)` feature layout.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


CLASS_NAMES_3WAY: Tuple[str, str, str] = ("Unbound_Field", "Potential_Escaper", "Bound_Core")


def compute_node_relational_features_v2(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    err_x: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    ptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute N-invariant local neighborhood relational features and append uncertainty channels.

    Parameters
    ----------
    x : torch.Tensor, shape (N, d)
        Standardized node observable vector (e.g., d=5 or d=4 subset).
    edge_index : torch.Tensor, shape (2, E)
        Directed k-NN graph edges `(src, dst)` built strictly on 2D tangent sky `(dRA*, dDec)`.
    edge_attr : torch.Tensor, shape (E, >=3), optional
        Physical edge attributes `[dRA*_ij, dDec_ij, theta_sky_ij]`. If None, falls
        back to Euclidean distance in `x[:, :2]`.
    err_x : torch.Tensor, shape (N, d_err), optional
        Bounded astrometric uncertainty and parallax-SNR channels.
    batch : torch.Tensor, shape (N,), optional
        Graph assignment index for PyG batches.
    ptr : torch.Tensor, shape (B+1,), optional
        Cumulative node offsets for PyG batches.

    Returns
    -------
    torch.Tensor, shape (N, 3*d + 3 + d_err)
        Concatenation `[x, x - <x>_N, sigma_N(x), <theta>_N, log_norm_rho, spread_ratio, (err_x)]`.
    """
    n_nodes, _ = x.shape
    src, dst = edge_index[0], edge_index[1]

    ones = torch.ones(dst.shape[0], dtype=x.dtype, device=x.device)
    deg = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(0, dst, ones)
    deg_safe = deg.clamp(min=1.0).unsqueeze(-1)

    x_sum = torch.zeros_like(x).index_add_(0, dst, x[src])
    x_mean = x_sum / deg_safe
    x_diff = x - x_mean

    sq_diff = (x[src] - x_mean[dst]).pow(2)
    x_var = torch.zeros_like(x).index_add_(0, dst, sq_diff) / deg_safe
    x_std = torch.sqrt(x_var + 1e-6)

    if edge_attr is not None and edge_attr.shape[1] >= 3:
        r_edge = edge_attr[:, 2].to(dtype=x.dtype)
    else:
        r_edge = torch.norm(x[src, : min(2, x.shape[1])] - x[dst, : min(2, x.shape[1])], dim=-1)

    r_sum = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(0, dst, r_edge)
    mean_r = r_sum / deg_safe.squeeze(-1)

    r_var = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(
        0, dst, (r_edge - mean_r[dst]).pow(2)
    ) / deg_safe.squeeze(-1)
    std_r = torch.sqrt(r_var + 1e-6)
    r_spread = std_r / (mean_r + 1e-4)

    # N-normalized local sky surface density: invariant to magnitude-limit thinning
    if ptr is not None and ptr.numel() >= 2:
        counts = ptr[1:] - ptr[:-1]
        n_graph = counts.repeat_interleave(counts).to(dtype=x.dtype, device=x.device)
    elif batch is not None:
        counts = torch.bincount(batch)
        n_graph = counts[batch].to(dtype=x.dtype, device=x.device)
    else:
        n_graph = torch.full((n_nodes,), float(max(n_nodes, 1)), dtype=x.dtype, device=x.device)

    norm_rho = deg.clamp(min=1.0) / (n_graph.clamp(min=1.0) * torch.pi * (mean_r.pow(2) + 1e-4))
    log_norm_rho = torch.log(norm_rho.clamp(min=1e-6)) / 10.0

    parts = [
        x,
        x_diff,
        x_std,
        mean_r.unsqueeze(-1),
        log_norm_rho.unsqueeze(-1),
        r_spread.unsqueeze(-1),
    ]
    if err_x is not None and err_x.numel() > 0:
        parts.append(err_x.to(dtype=x.dtype))

    return torch.cat(parts, dim=-1)


class GAT_V2(nn.Module):
    """3-layer Multi-Head Residual Graph Attention Network for Gaia DR3 V2 catalogs."""

    def __init__(
        self,
        in_features: int = 5,
        err_features: int = 4,
        hidden_channels: int = 64,
        heads: int = 4,
        out_channels: int = 3,
        edge_dim: int = 3,
        dropout: float = 0.2,
        edge_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.err_features = int(err_features)
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)
        self.dropout = float(dropout)
        self.edge_dropout = float(min(self.dropout, 0.15) if edge_dropout is None else edge_dropout)

        if hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})."
            )
        head_dim = hidden_channels // heads
        rel_dim = 3 * self.in_features + 3 + self.err_features
        self.rel_dim = rel_dim

        self.lin_in = nn.Sequential(
            nn.Linear(rel_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
        )

        conv_edge_dim = self.edge_dim if self.edge_dim > 0 else None
        self.conv1 = GATConv(hidden_channels, head_dim, heads=heads, dropout=self.edge_dropout, edge_dim=conv_edge_dim)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GATConv(hidden_channels, head_dim, heads=heads, dropout=self.edge_dropout, edge_dim=conv_edge_dim)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # Final message-passing layer: keep attention edge_dropout=0.0 to preserve neighborhood aggregation
        self.conv3 = GATConv(hidden_channels, head_dim, heads=heads, dropout=0.0, edge_dim=conv_edge_dim)
        self.norm3 = nn.LayerNorm(hidden_channels)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        err_x: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing per-star class logits of shape (N, out_channels)."""
        if h0 is not None and h0.shape[0] == x.shape[0] and h0.shape[1] == self.rel_dim:
            h = h0
        else:
            h = compute_node_relational_features_v2(
                x,
                edge_index,
                edge_attr=edge_attr,
                err_x=err_x,
                batch=batch,
                ptr=ptr,
            )
        h = self.lin_in(h)

        conv_attr = edge_attr if (edge_attr is not None and self.edge_dim > 0) else None
        for conv, norm in (
            (self.conv1, self.norm1),
            (self.conv2, self.norm2),
            (self.conv3, self.norm3),
        ):
            delta = conv(h, edge_index, edge_attr=conv_attr)
            h = h + F.dropout(F.elu(norm(delta)), p=self.dropout, training=self.training)

        return self.head(h)


def load_pretrained_with_channel_adaptation_v2(
    model: GAT_V2,
    state_dict: Dict[str, torch.Tensor],
    selected_features: Optional[Sequence[int]] = None,
    d_err: Optional[int] = None,
) -> Tuple[int, int]:
    """Load pre-trained V2 weights onto `model`, adapting `lin_in.0.weight` for 5D -> 4D transfer.

    Preserves exact block alignment `(x, x - <x>, sigma(x), sky_scalars[3], err_x[d_err])`
    and rescales adapted input weights by `sqrt(old_dim / new_dim)` to conserve signal variance.

    Returns
    -------
    tuple[int, int]
        `(n_loaded_tensors, n_skipped_tensors)`.
    """
    clean_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        clean_key = k[len("module.") :] if k.startswith("module.") else k
        clean_state[clean_key] = v

    target_state = model.state_dict()
    err_dim = model.err_features if d_err is None else int(d_err)

    in_key = "lin_in.0.weight"
    if in_key in clean_state and in_key in target_state:
        old_w = clean_state[in_key]
        new_w = target_state[in_key]
        if old_w.shape != new_w.shape and selected_features is not None:
            old_dim = int(old_w.shape[1])
            d_old = (old_dim - 3 - err_dim) // 3
            if 3 * d_old + 3 + err_dim == old_dim and len(selected_features) < d_old:
                sel = [int(i) for i in selected_features]
                keep_idx = (
                    sel
                    + [d_old + i for i in sel]
                    + [2 * d_old + i for i in sel]
                    + [3 * d_old, 3 * d_old + 1, 3 * d_old + 2]
                    + [3 * d_old + 3 + j for j in range(err_dim)]
                )
                if len(keep_idx) == new_w.shape[1] and max(keep_idx) < old_dim:
                    scale = (float(old_dim) / float(len(keep_idx))) ** 0.5
                    clean_state[in_key] = old_w[:, keep_idx].clone() * scale

    n_loaded = 0
    n_skipped = 0
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in clean_state.items():
        if k in target_state and target_state[k].shape == v.shape:
            filtered[k] = v
            n_loaded += 1
        else:
            n_skipped += 1

    model.load_state_dict(filtered, strict=False)
    return n_loaded, n_skipped
