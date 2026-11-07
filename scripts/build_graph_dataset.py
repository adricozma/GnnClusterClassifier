"""Construct PyTorch Geometric k-NN stellar stream graphs from extracted observables.

Supports multi-core CPU parallelization (`--num_workers`) and arbitrary observable subsets
(`--feature_indices`, e.g., `0,1,2,3,4` for 5D astrometry or 4-observable combinations for
ablation studies). Uses an O(N log N) KD-tree (`scipy.spatial.cKDTree`) to construct directed
k-NN graphs (`source_to_target`, `loop=False`).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_feature_indices(indices_str):
    """Parse a comma-separated string of column indices (e.g. '0,1,2,3,4')."""
    return [int(x.strip()) for x in indices_str.split(",") if x.strip() != ""]


def build_knn_edge_index(features, k=20):
    """Build a directed k-NN graph edge_index (source_to_target, loop=False) via cKDTree."""
    n_nodes = features.shape[0]
    k_eff = min(k, max(1, n_nodes - 1))
    tree = cKDTree(features)
    _, neighbors = tree.query(features, k=k_eff + 1)
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
    feature_indices=None,
):
    """Generate and save PyTorch Geometric Data objects for a single simulation."""
    if snapshot_list is None:
        snapshot_list = list(range(0, 101))
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3, 4]

    os.makedirs(output_folder_path, exist_ok=True)
    saved_count = 0

    for snap in snapshot_list:
        data_path = os.path.join(data_folder_path, f"obs_matrices_snapshot_{snap:04d}.npy")
        label_path = os.path.join(labels_folder_path, f"B_L_snapshot_{snap:04d}.npy")

        if not (os.path.exists(data_path) and os.path.exists(label_path)):
            continue

        all_rot_exist = all(
            os.path.exists(
                os.path.join(output_folder_path, f"graph_data_output_{snap}_rotation_{r}.pt")
            )
            and os.path.exists(
                os.path.join(output_folder_path, f"label_output_{snap}_rotation_{r}.pt")
            )
            for r in range(n_rotations)
        )
        if all_rot_exist:
            saved_count += n_rotations
            continue

        data_matrix3d = np.load(data_path)
        label_matrix = np.load(label_path)
        label_tensor = torch.tensor(label_matrix, dtype=torch.float)

        for rot_idx in range(min(n_rotations, data_matrix3d.shape[0])):
            graph_out_file = os.path.join(
                output_folder_path, f"graph_data_output_{snap}_rotation_{rot_idx}.pt"
            )
            label_out_file = os.path.join(
                output_folder_path, f"label_output_{snap}_rotation_{rot_idx}.pt"
            )
            features_sub = np.asarray(
                data_matrix3d[rot_idx][:, feature_indices], dtype=np.float32
            )
            edge_index = build_knn_edge_index(features_sub, k=k_neighbors)
            graph_data = Data(
                x=torch.from_numpy(features_sub),
                edge_index=edge_index,
            )

            torch.save(graph_data, graph_out_file)
            torch.save(label_tensor, label_out_file)
            saved_count += 1

    return os.path.basename(output_folder_path), saved_count


def _worker_build_ic(task):
    return load_data_and_create_graphs(*task)


def main():
    parser = argparse.ArgumentParser(
        description="Build k-NN PyTorch Geometric graph dataset from extracted observables."
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "obs_clean"),
        help="Input directory containing per-IC observable matrices.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "labels_clean"),
        help="Input directory containing per-IC bound label arrays.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/cozma/graph_data_clean",
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
        help="Number of nearest neighbors in feature space (default: 20).",
    )
    parser.add_argument(
        "--feature_indices",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated observable column indices to include (0:RA, 1:Dec, 2:Parallax, 3:pmRA, 4:pmDec, 5:vlos). Default: '0,1,2,3,4'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK", os.environ.get("SLURM_NTASKS", 16)
            )
        ),
        help="Number of parallel CPU worker processes across IC folders.",
    )
    args = parser.parse_args()

    feat_idx = parse_feature_indices(args.feature_indices)
    os.makedirs(args.output_dir, exist_ok=True)
    ic_dirs = sorted(
        [
            d
            for d in os.listdir(args.obs_dir)
            if d.startswith("IC_") and os.path.isdir(os.path.join(args.obs_dir, d))
        ],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )

    print(
        f"Building graphs for {len(ic_dirs)} ICs (features={feat_idx}, k={args.k_neighbors}) "
        f"into '{args.output_dir}' with {args.num_workers} CPU workers..."
    )

    tasks = [
        (
            os.path.join(args.obs_dir, ic_name),
            os.path.join(args.labels_dir, ic_name),
            os.path.join(args.output_dir, ic_name),
            args.rotations,
            None,
            args.k_neighbors,
            feat_idx,
        )
        for ic_name in ic_dirs
    ]

    if args.num_workers <= 1:
        for t in tasks:
            ic_name, cnt = _worker_build_ic(t)
            print(f"Completed {ic_name}: {cnt} graphs saved.")
    else:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, max(1, len(tasks)))) as pool:
            futures = [pool.submit(_worker_build_ic, t) for t in tasks]
            for fut in as_completed(futures):
                ic_name, cnt = fut.result()
                print(f"Completed {ic_name}: {cnt} graphs saved.")


if __name__ == "__main__":
    main()
