"""Build Study 2 V2 PyTorch Geometric 2D-sky k-NN graphs from extracted Gaia V2 observables.

Constructs edges strictly in 2D tangent sky space (dRA*, dDec) so that graph topology
never distorts with Gaia DR3 parallax uncertainties or negative parallax measurements.
Saves per-node astrometry `x`, per-star error/SNR features `err_x`, 2D physical edge
offsets `[dRA*, dDec, theta_sky]` as `edge_attr`, and 3-class labels `y` (`0`: Unbound/Field,
`1`: Potential Escaper, `2`: Bound Core).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_feature_indices(raw: str) -> List[int]:
    if not raw.strip():
        return [0, 1, 2, 3, 4]
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def build_sky_knn_edges(
    sky_2d: np.ndarray,
    k_neighbors: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build directed k-NN edges and physical [dRA*, dDec, theta_sky] edge vectors."""
    n_nodes = len(sky_2d)
    k_eff = min(int(k_neighbors), max(n_nodes - 1, 1))
    tree = cKDTree(sky_2d)
    dists, idxs = tree.query(sky_2d, k=k_eff + 1)

    if k_eff == 1:
        dists = dists.reshape(n_nodes, -1)
        idxs = idxs.reshape(n_nodes, -1)

    src = idxs[:, 1:].reshape(-1).astype(np.int64)
    dst = np.repeat(np.arange(n_nodes, dtype=np.int64), k_eff)
    edge_index = np.vstack((src, dst))

    delta_sky = (sky_2d[src] - sky_2d[dst]).astype(np.float32)
    r_sky = dists[:, 1:].reshape(-1, 1).astype(np.float32)
    edge_attr = np.hstack((delta_sky, r_sky)).astype(np.float32)
    return edge_index, edge_attr


def _is_valid_pt(path: str, min_bytes: int = 512) -> bool:
    return bool(os.path.isfile(path) and os.path.getsize(path) >= min_bytes)


def _worker_build_ic(task: Tuple) -> Tuple[str, int, int]:
    (
        ic_obs_dir,
        ic_out_dir,
        ic_name,
        regime,
        k_neighbors,
        feat_indices,
        overwrite,
    ) = task

    os.makedirs(ic_out_dir, exist_ok=True)
    npz_files = sorted(
        f for f in os.listdir(ic_obs_dir) if f.startswith("v2_obs_") and f.endswith(".npz")
    )

    built = 0
    skipped = 0
    x_prefix = f"{regime}_x_"
    err_prefix = f"{regime}_err_"

    for fname in npz_files:
        snap_stem = fname[len("v2_obs_") : -len(".npz")]
        npz_path = os.path.join(ic_obs_dir, fname)
        with np.load(npz_path) as data:
            n_rot = int(data["n_rotations"])
            for r in range(n_rot):
                out_pt = os.path.join(ic_out_dir, f"graph_v2_{snap_stem}_rot_{r}.pt")
                if not overwrite and _is_valid_pt(out_pt):
                    skipped += 1
                    continue

                x_full = np.asarray(data[f"{x_prefix}{r}"], dtype=np.float32)
                err_x = np.asarray(data[f"{err_prefix}{r}"], dtype=np.float32)
                y_arr = np.asarray(data[f"y_{r}"], dtype=np.int64)
                is_field = np.asarray(data[f"is_field_{r}"], dtype=bool)

                if len(x_full) < 2:
                    continue

                sky_2d = np.asarray(x_full[:, :2], dtype=np.float64)
                edge_index_np, edge_attr_np = build_sky_knn_edges(sky_2d, k_neighbors=k_neighbors)
                x_sel = np.ascontiguousarray(x_full[:, feat_indices], dtype=np.float32)

                graph = Data(
                    x=torch.from_numpy(x_sel),
                    err_x=torch.from_numpy(np.ascontiguousarray(err_x, dtype=np.float32)),
                    edge_index=torch.from_numpy(edge_index_np),
                    edge_attr=torch.from_numpy(edge_attr_np),
                    y=torch.from_numpy(y_arr),
                    is_field=torch.from_numpy(is_field),
                )

                tmp_pt = out_pt + f".tmp.{os.getpid()}.pt"
                torch.save(graph, tmp_pt)
                os.replace(tmp_pt, out_pt)
                built += 1

    return ic_name, built, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Study 2 V2 2D-sky k-NN PyG graphs from V2 `.npz` observables."
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        default="/scratch/cozma/study2_v2/obs_unbiased",
        help="Input directory containing IC_*/v2_obs_*.npz.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/cozma/study2_v2/graphs_noisy",
        help="Output directory for IC_*/graph_v2_*.pt.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="noisy",
        choices=["clean", "noisy"],
        help="Select 'clean' or 'noisy' realizations from the `.npz` files.",
    )
    parser.add_argument("--k_neighbors", type=int, default=32)
    parser.add_argument(
        "--feature_indices",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated column indices to include in Data.x (default 5D astrometry: 0,1,2,3,4).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_NTASKS", 16))),
    )
    args = parser.parse_args()

    feat_indices = parse_feature_indices(args.feature_indices)
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
        f"Building V2 {args.regime} sky-kNN graphs (k={args.k_neighbors}, features={feat_indices}) "
        f"across {len(ic_dirs)} IC folder(s) into '{args.output_dir}'..."
    )

    tasks = [
        (
            os.path.join(args.obs_dir, ic_name),
            os.path.join(args.output_dir, ic_name),
            ic_name,
            args.regime,
            args.k_neighbors,
            feat_indices,
            args.overwrite,
        )
        for ic_name in ic_dirs
    ]

    total_built = 0
    total_skipped = 0
    if args.num_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            _, b, s = _worker_build_ic(t)
            total_built += b
            total_skipped += s
    else:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(tasks))) as pool:
            futs = [pool.submit(_worker_build_ic, t) for t in tasks]
            for fut in as_completed(futs):
                _, b, s = fut.result()
                total_built += b
                total_skipped += s

    print(f"V2 graph build complete: {total_built} generated, {total_skipped} skipped (cached).")


if __name__ == "__main__":
    main()
