"""Construct PyTorch Geometric 5D k-NN stellar stream graphs from extracted observables.

For each simulation directory `IC_<id>`, snapshot `0000..0100`, and 3D rotation `0..n-1`,
this script loads the 5D astrometric feature matrix (excluding line-of-sight radial
velocity v_los) and corresponding Küpper et al. (2010a,b) bound labels, builds a
directed k-NN graph (k=20 or k=30, loop=False) using an O(N log N) 5D KD-tree, and
exports `graph_data_output_<snap>_rotation_<rot>.pt` and `label_output_<snap>_rotation_<rot>.pt`.
"""

import argparse
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_knn_edge_index(features_5d, k=20):
    """Build a directed k-NN graph edge_index (source_to_target, loop=False) via cKDTree.

    Matches `torch_geometric.nn.knn_graph(x, k=k, loop=False, flow='source_to_target')`
    while running in O(N log N) time on CPU instead of O(N^2).
    """
    n_nodes = features_5d.shape[0]
    k_eff = min(k, max(1, n_nodes - 1))
    tree = cKDTree(features_5d)
    _, neighbors = tree.query(features_5d, k=k_eff + 1)
    source_nodes = neighbors[:, 1:].reshape(-1)
    target_nodes = np.repeat(np.arange(n_nodes, dtype=np.int64), k_eff)
    edge_index = np.vstack((source_nodes, target_nodes))
    return torch.from_numpy(edge_index).long()


def load_data_and_create_graphs(
    data_folder_path,
    labels_folder_path,
    output_folder_path,
    n_rotations=5,
    snapshot_list=None,
    k_neighbors=20,
):
    """Generate and save PyTorch Geometric Data objects for a single simulation."""
    if snapshot_list is None:
        snapshot_list = list(range(0, 101))

    os.makedirs(output_folder_path, exist_ok=True)

    for snap in snapshot_list:
        data_path = os.path.join(data_folder_path, f"obs_matrices_snapshot_{snap:04d}.npy")
        label_path = os.path.join(labels_folder_path, f"B_L_snapshot_{snap:04d}.npy")

        if not (os.path.exists(data_path) and os.path.exists(label_path)):
            continue

        data_matrix3d = np.load(data_path)
        label_matrix = np.load(label_path)
        label_tensor = torch.tensor(label_matrix, dtype=torch.float)

        for rot_idx in range(min(n_rotations, data_matrix3d.shape[0])):
            data_matrix_5d = np.asarray(data_matrix3d[rot_idx][:, :5], dtype=np.float32)
            edge_index = build_knn_edge_index(data_matrix_5d, k=k_neighbors)
            graph_data = Data(
                x=torch.from_numpy(data_matrix_5d),
                edge_index=edge_index,
            )

            torch.save(
                graph_data,
                os.path.join(output_folder_path, f"graph_data_output_{snap}_rotation_{rot_idx}.pt"),
            )
            torch.save(
                label_tensor,
                os.path.join(output_folder_path, f"label_output_{snap}_rotation_{rot_idx}.pt"),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build 5D k-NN PyTorch Geometric graph dataset from extracted observables."
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "obs_all_stars"),
        help="Input directory containing per-IC observable matrices.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "labels_bound_stars"),
        help="Input directory containing per-IC bound label arrays.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "graph_data"),
        help="Output directory for PyTorch Geometric .pt files.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=5,
        help="Number of 3D rotations per snapshot (default: 5).",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=20,
        help="Number of nearest neighbors in 5D feature space (default: 20).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ic_dirs = sorted(
        d for d in os.listdir(args.obs_dir)
        if d.startswith("IC_") and os.path.isdir(os.path.join(args.obs_dir, d))
    )

    for ic_name in ic_dirs:
        ic_obs_path = os.path.join(args.obs_dir, ic_name)
        ic_labels_path = os.path.join(args.labels_dir, ic_name)
        ic_out_path = os.path.join(args.output_dir, ic_name)
        print(f"Building graphs for {ic_name} (k={args.k_neighbors})...")
        load_data_and_create_graphs(
            data_folder_path=ic_obs_path,
            labels_folder_path=ic_labels_path,
            output_folder_path=ic_out_path,
            n_rotations=args.rotations,
            k_neighbors=args.k_neighbors,
        )


if __name__ == "__main__":
    main()
