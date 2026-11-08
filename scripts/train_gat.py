"""Training and curriculum fine-tuning script for the 3-layer Multi-Head Residual GAT classifier.

Supports:
- Simulation-level (`--split_mode ic`) train/val/test partitioning (zero data leakage).
- Multi-node CPU DistributedDataParallel (DDP) across SLURM nodes (`torchrun` / `srun`).
- Micro-batch gradient accumulation preserving exact effective batch size (`--bs 64` via `--micro_bs 8`).
- Automatic disk activation offloading (`DiskActivationOffloader`) to `/scratch` when RAM approaches cgroup limit.
- Two-stage Curriculum Learning and 5D -> 4D channel-adapted transfer (`--pretrained_weights`).
- Real-time epoch-by-epoch live training curve generation (`scripts/live_plot_utils.py`).
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from distributed_trainer import run_training


def build_parser():
    parser = argparse.ArgumentParser(description="Train or fine-tune the 3-layer Multi-Head Residual GAT model.")
    parser.add_argument("--data_folder", "--data_dir", dest="data_folder", type=str, required=True, help="Path to graph data folder.")
    parser.add_argument("--n", type=int, default=5, help="Number of 3D rotations per snapshot.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=64, help="Effective global mini-batch size.")
    parser.add_argument("--micro_bs", type=int, default=8, help="Max graphs processed per forward/backward micro-step per node.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default="training_metrics_gat.log", help="Log file path.")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="ic",
        choices=["ic", "file", "random"],
        help="Split dataset by stream ID ('ic', no data leakage) or by individual file ('file', legacy).",
    )
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight for BCEWithLogitsLoss.")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="",
        help="Optional path to pre-trained model weights (.pth) for curriculum fine-tuning or 5D->4D transfer.",
    )
    parser.add_argument("--save_model", type=str, default="best_gat_model.pth", help="Output path to save the best model checkpoint (.pth).")
    parser.add_argument(
        "--feature_indices",
        type=str,
        default="",
        help="Optional comma-separated feature column indices to select on the fly (e.g. '0,1,3,4' for 4-observable ablation).",
    )
    parser.add_argument("--force_disk_offload", action="store_true", help="Force offloading intermediate autograd tensors to /scratch.")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable automatic epoch checkpoint resume.")
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_training("gat", args)
