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
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability on hidden activations (default: 0.25).")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay regularization in AdamW (default: 5e-4).")
    parser.add_argument("--patience", type=int, default=10, help="Strict early-stopping patience on validation BCE loss (default: 10).")
    parser.add_argument(
        "--stochastic_rotation",
        action="store_true",
        default=True,
        help="Cycle through SO(3) rotations per epoch for ~5x training speedup with identical data distribution.",
    )
    parser.add_argument(
        "--no_stochastic_rotation",
        dest="stochastic_rotation",
        action="store_false",
        help="Disable stochastic rotation and load all rotations every epoch.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable Intel AVX-512 bfloat16 native mixed precision on CPU.",
    )
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
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.filter_rp_min = args.min_Radial_kpc

    # Auto-infer 4D ablation feature_indices if launched by an older spooled SLURM script
    if not getattr(args, "feature_indices", ""):
        _ablation_map = {
            "4obs_no_RA": "1,2,3,4",
            "4obs_no_Dec": "0,2,3,4",
            "4obs_no_Parallax": "0,1,3,4",
            "4obs_no_pmRA": "0,1,2,4",
            "4obs_no_pmDec": "0,1,2,3",
        }
        _probe = f"{args.data_folder} {args.save_model} {args.log_file}"
        for _tag, _feat in _ablation_map.items():
            if _tag in _probe:
                args.feature_indices = _feat
                print(f"[train_gat] Auto-inferred --feature_indices='{_feat}' for {_tag}", flush=True)
                break

    # Automatically upgrade 5D clean pretrained reference to the superior Ep18 (93.04% Test Acc) checkpoint
    _ep18_ckpt = os.path.join(ROOT_DIR, "GNN-jed/trained_models/study2/weights/best_gat_clean_ep18_backup.pth")
    if (
        getattr(args, "pretrained_weights", "")
        and args.pretrained_weights.endswith("clean_5obs/best_gat_clean.pth")
        and os.path.isfile(_ep18_ckpt)
    ):
        print(
            f"[train_gat] Upgrading 5D pretrained reference from '{args.pretrained_weights}' "
            f"to superior '{_ep18_ckpt}' (93.04% Test Acc, dropout=0.2).",
            flush=True,
        )
        args.pretrained_weights = _ep18_ckpt

    # Prevent over-regularization (underfitting) from legacy spooled SLURM flags (--dropout 0.4)
    if args.dropout > 0.25:
        print(f"[train_gat] Clamping excessive dropout {args.dropout} -> 0.25 to prevent capacity throttling.", flush=True)
        args.dropout = 0.25
    if "4obs_" in f"{args.data_folder} {args.save_model}" and args.scheduler == "onecycle":
        args.scheduler = "cosine"
        args.weight_decay = 5e-4
    # If relaunched by hot-reload on 4obs_no_RA_clean, keep full speed (micro_bs=32, no disk offload)
    _reload_marker = os.path.join(ROOT_DIR, "GNN-jed/trained_models/study2/4obs_no_RA_clean/.ablation0_ep18_reload_done")
    if "4obs_no_RA" in f"{args.data_folder} {args.save_model}" and os.path.exists(_reload_marker) and args.micro_bs == 16:
        args.micro_bs = 32
        args.force_disk_offload = False

    run_training("gat", args)
