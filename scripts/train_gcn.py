"""Training and curriculum fine-tuning script for the 3-layer Residual GCN classifier.

Supports:
- Simulation-level (`--split_mode ic`) train/val/test partitioning (zero data leakage).
- Multi-node CPU DistributedDataParallel (DDP) across SLURM nodes (`torchrun` / `srun`).
- Micro-batch gradient accumulation preserving exact effective batch size (`--bs 256` via `--micro_bs 32`).
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
    parser = argparse.ArgumentParser(description="Train or fine-tune the 3-layer Residual GCN model.")
    parser.add_argument("--data_folder", "--data_dir", dest="data_folder", type=str, required=True, help="Path to graph data folder.")
    parser.add_argument("--n", type=int, default=5, help="Number of 3D rotations per snapshot.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=256, help="Effective global mini-batch size.")
    parser.add_argument("--micro_bs", type=int, default=32, help="Max graphs processed per forward/backward micro-step per node.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default="training_metrics_gcn.log", help="Log file path.")
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
    parser.add_argument("--save_model", type=str, default="best_gcn_model.pth", help="Output path to save the best model checkpoint (.pth).")
    parser.add_argument(
        "--feature_indices",
        type=str,
        default="",
        help="Optional comma-separated feature column indices to select on the fly (e.g. '0,1,3,4' for 4-observable ablation).",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability on hidden activations (default: 0.2).")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 weight decay regularization in AdamW (default: 1e-3).")
    parser.add_argument("--patience", type=int, default=4, help="Strict early-stopping patience on validation BCE loss (default: 4).")
    parser.add_argument(
        "--min_Radial_kpc",
        "--filter_rp_min",
        dest="min_Radial_kpc",
        type=float,
        default=0.0,
        help="Optional minimum perigalacticon R_p [kpc] filter on streams (e.g. 5.0).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["onecycle", "cosine", "plateau"],
        help="Adaptive learning rate scheduler.",
    )
    parser.add_argument("--force_disk_offload", action="store_true", help="Force offloading intermediate autograd tensors to /scratch.")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable automatic epoch checkpoint resume.")
    parser.add_argument("--allow_non_baseline_gcn", action="store_true", help="Allow running GCN outside the 5D noiseless baseline (disabled by default to save compute time).")
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.filter_rp_min = args.min_Radial_kpc
    save_basename = os.path.basename(args.save_model)
    if not args.allow_non_baseline_gcn and save_basename not in ("best_gcn_clean.pth", "best_gcn_model.pth"):
        print(
            f"[SKIP] GCN training is restricted to the 5-observable noiseless baseline (clean_5obs) "
            f"to demonstrate the architectural comparison. Skipping '{save_basename}' (GAT-only regime)."
        )
        sys.exit(0)
    run_training("gcn", args)
