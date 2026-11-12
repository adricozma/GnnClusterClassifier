""" Unified distributed / gradient-accumulation / disk-offload training engine for GCN and GAT."""

import gc
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler, random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_models import (
    GAT,
    GCN,
    compute_node_relational_features,
    load_pretrained_with_channel_adaptation,
)
from live_plot_utils import parse_log_file, update_live_plots_for_log
from oom_Pin_and_offload import (
    DiskActivationOffloader,
    get_memory_limit_gb,
    get_process_and_cgroup_rss_gb,
    init_distributed,
)

CENTRAL_WEIGHTS_DIR = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "weights")


def _atomic_save_state_dict(state_dict: dict, target_path: str) -> None:
    """Atomically write a PyTorch state_dict to disk so interrupted I/O never corrupts weights."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, target_path)


def _archive_and_register_weights(
    best_model_path: str,
    arch: str,
    in_channels: int,
    epoch: int,
    val_loss: float,
    val_acc: float,
    test_loss: float | None = None,
    test_acc: float | None = None,
    test_f1: float | None = None,
    val_core_fnr: float | None = None,
    test_core_fnr: float | None = None,
) -> None:
    """Mirror saved model weights into the central `GNN-jed/trained_models/study2/weights/` directory and update registry."""
    if not os.path.exists(best_model_path):
        return
    os.makedirs(CENTRAL_WEIGHTS_DIR, exist_ok=True)
    fname = os.path.basename(best_model_path)
    central_copy = os.path.join(CENTRAL_WEIGHTS_DIR, fname)
    if os.path.abspath(best_model_path) != os.path.abspath(central_copy):
        tmp_copy = central_copy + ".tmp"
        shutil.copyfile(best_model_path, tmp_copy)
        os.replace(tmp_copy, central_copy)

    manifest_path = os.path.join(CENTRAL_WEIGHTS_DIR, "weights_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    rel_exp_path = os.path.relpath(best_model_path, ROOT_DIR)
    rel_central_path = os.path.relpath(central_copy, ROOT_DIR)
    size_kb = os.path.getsize(best_model_path) / 1024.0
    prev = manifest.get(fname, {})

    manifest[fname] = {
        "filename": fname,
        "architecture": arch.upper(),
        "in_channels": in_channels,
        "experiment_path": rel_exp_path,
        "central_archive_path": rel_central_path,
        "size_kb": round(size_kb, 1),
        "best_epoch": int(epoch),
        "val_loss": round(float(val_loss), 4),
        "val_acc": round(float(val_acc), 2),
        "val_core_fnr": round(float(val_core_fnr), 2) if val_core_fnr is not None else prev.get("val_core_fnr"),
        "test_loss": round(float(test_loss), 4) if test_loss is not None else prev.get("test_loss"),
        "test_acc": round(float(test_acc), 2) if test_acc is not None else prev.get("test_acc"),
        "test_f1": round(float(test_f1), 4) if test_f1 is not None else prev.get("test_f1"),
        "test_core_fnr": round(float(test_core_fnr), 2) if test_core_fnr is not None else prev.get("test_core_fnr"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    registry_md = os.path.join(CENTRAL_WEIGHTS_DIR, "MODEL_WEIGHTS_REGISTRY.md")
    lines = [
        "# Study 2 Trained GNN Weights Registry",
        "",
        "All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.",
        "",
        "| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Experiment Path |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for k in sorted(manifest.keys()):
        m = manifest[k]
        t_acc = f"{m['test_acc']:.2f}%" if m.get("test_acc") is not None else "Pending"
        t_f1 = f"{m['test_f1']:.4f}" if m.get("test_f1") is not None else "-"
        c_fnr_val = m.get("test_core_fnr") if m.get("test_core_fnr") is not None else m.get("val_core_fnr")
        c_fnr = f"{c_fnr_val:.2f}%" if c_fnr_val is not None else "-"
        lines.append(
            f"| `{m['filename']}` | **{m['architecture']}** | {m['in_channels']}D | {m['best_epoch']} | "
            f"{m['val_loss']:.4f} | **{m['val_acc']:.2f}%** | {c_fnr} | **{t_acc}** | {t_f1} | "
            f"{m['size_kb']:.1f} KB | `{m['experiment_path']}` |"
        )
    with open(registry_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class AstroGraphDataset(Dataset):
    def __init__(self, file_paths, feat_idx=None):
        self.file_paths = file_paths
        self.feat_idx = feat_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        data = torch.load(filepath, weights_only=False)
        if self.feat_idx is not None and data.x.shape[1] > len(self.feat_idx):
            data.x = data.x[:, self.feat_idx]

        label_path = filepath.replace("graph_data_output", "label_output")
        target = torch.load(label_path, weights_only=False)
        data.y = target.float()
        with torch.no_grad():
            data.h0 = compute_node_relational_features(data.x, data.edge_index)
        return data


def _collate_as_list(data_list):
    """Return a plain Python list of PyG Data objects so micro-batches can be sliced without duplicating memory."""
    return data_list


def run_training(arch: str, args):
    is_ddp, rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    data_folder = args.data_folder
    n = args.n
    Lr = args.lr
    effective_bs = args.bs
    epochs = args.epochs
    log_filename = args.log_file
    split_mode = args.split_mode
    pos_weight_val = args.pos_weight
    pretrained_weights = args.pretrained_weights
    best_model_path = args.save_model
    feature_indices = (
        [int(x.strip()) for x in args.feature_indices.split(",") if x.strip() != ""]
        if getattr(args, "feature_indices", "")
        else None
    )

    default_micro = 32 if arch == "gcn" else 16
    micro_bs = getattr(args, "micro_bs", 0) or default_micro
    dropout_val = float(getattr(args, "dropout", 0.4 if arch == "gat" else 0.2))
    weight_decay_val = float(getattr(args, "weight_decay", 1e-3 if arch == "gat" else 1e-4))
    scheduler_type = str(getattr(args, "scheduler", "onecycle")).lower()
    filter_rp_min = float(getattr(args, "filter_rp_min", getattr(args, "min_Radial_kpc", 0.0)))

    if is_main:
        os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)
        os.makedirs(CENTRAL_WEIGHTS_DIR, exist_ok=True)

    # ---------- CPU / DEVICE MANAGEMENT ----------
    cpus_per_task = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            os.environ.get("OMP_NUM_THREADS", os.environ.get("SLURM_NTASKS", os.cpu_count() or 36)),
        )
    )
    math_threads = min(max(1, cpus_per_task), 72)
    torch.set_num_threads(math_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_rank_bs = max(1, effective_bs // world_size)
    mem_limit_gb = get_memory_limit_gb(246.0)
    if mem_limit_gb >= 120.0 and arch == "gat" and micro_bs < 16:
        micro_bs = min(per_rank_bs, 16)
    micro_bs = min(micro_bs, per_rank_bs)
    eval_bs = min(per_rank_bs, max(micro_bs, 32))

    # Strict validation-loss early stopping to prevent overfitting on surviving dense cores.
    default_patience = 5 if arch == "gcn" else 10
    early_stop_patience = int(getattr(args, "patience", default_patience) or default_patience)
    if arch == "gcn":
        epochs = min(epochs, 25)
    else:
        epochs = min(epochs, 50)

    if is_main:
        print(
            f"[{arch.upper()}] Device: {device} | DDP World Size: {world_size} node(s) | "
            f"Threads/rank: {torch.get_num_threads()} | RAM Limit/node: {mem_limit_gb:.1f} GB"
        )
        print(
            f"[{arch.upper()}] Effective Global Batch Size: {effective_bs} -> "
            f"{per_rank_bs}/node (Micro-batch size: {micro_bs} graphs, "
            f"{math.ceil(per_rank_bs / micro_bs)} grad-accum steps/node, Max Epochs: {epochs}, "
            f"Strict Val-Loss Patience: {early_stop_patience}, Dropout: {dropout_val}, "
            f"Weight Decay: {weight_decay_val:.1e}, Scheduler: {scheduler_type})"
        )

    outputlist = list(range(0, 101))
    if is_main:
        print(f"Indexing files in '{data_folder}' (split_mode='{split_mode}')...")

    ic_dirs = sorted(
        [
            d
            for d in os.listdir(data_folder)
            if d.startswith("IC_") and os.path.isdir(os.path.join(data_folder, d))
        ],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )

    if filter_rp_min > 0.0:
        import csv
        summary_csv = os.path.join(ROOT_DIR, "make_ICs", "unbiased_orbits_summary.csv")
        if os.path.exists(summary_csv):
            valid_sids = set()
            with open(summary_csv, "r", encoding="utf-8") as cf:
                for row in csv.DictReader(cf):
                    if float(row.get("R_peri_kpc", 0.0)) >= filter_rp_min:
                        valid_sids.add(int(row["Stream_ID"]))
            before_cnt = len(ic_dirs)
            ic_dirs = [
                d for d in ic_dirs
                if d.split("_")[1].isdigit() and int(d.split("_")[1]) in valid_sids
            ]
            if is_main:
                print(
                    f"Filtered streams by R_p >= {filter_rp_min:.2f} kpc: "
                    f"retained {len(ic_dirs)}/{before_cnt} streams."
                )

    def collect_files_for_ics(selected_ics):
        paths = []
        for ic_dir in selected_ics:
            data_folder_path = os.path.join(data_folder, ic_dir)
            for output in outputlist:
                for i in range(n):
                    graph_file = f"{data_folder_path}/graph_data_output_{output}_rotation_{i}.pt"
                    if os.path.exists(graph_file):
                        paths.append(graph_file)
        return paths

    if split_mode == "ic":
        train_ics_subset, val_ics_subset, test_ics_subset = random_split(
            ic_dirs, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )
        train_ics = [ic_dirs[i] for i in train_ics_subset.indices]
        val_ics = [ic_dirs[i] for i in val_ics_subset.indices]
        test_ics = [ic_dirs[i] for i in test_ics_subset.indices]

        if is_main:
            print(f"Stream-level split: {len(train_ics)} Train ICs, {len(val_ics)} Val ICs, {len(test_ics)} Test ICs")
            print(f"  Val ICs : {val_ics}")
            print(f"  Test ICs: {test_ics}")

        train_set = AstroGraphDataset(collect_files_for_ics(train_ics), feat_idx=feature_indices)
        val_set = AstroGraphDataset(collect_files_for_ics(val_ics), feat_idx=feature_indices)
        test_set = AstroGraphDataset(collect_files_for_ics(test_ics), feat_idx=feature_indices)
    else:
        file_paths = collect_files_for_ics(ic_dirs)
        if is_main:
            print(f"End of indexing! {len(file_paths)} files found.")
        dataset = AstroGraphDataset(file_paths, feat_idx=feature_indices)
        train_set, val_set, test_set = random_split(
            dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

    sample_graph = train_set[0]
    in_channels = int(sample_graph.x.shape[1])
    if is_main:
        print(f"Detected input node feature dimension: in_channels = {in_channels}")

    # Keep DataLoader worker count conservative (4 workers, prefetch_factor=1) to prevent worker RAM bloat
    nb_workers = min(4, max(2, math_threads // 16))

    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
        if is_ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )
    test_sampler = (
        DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )

    from torch.utils.data import DataLoader as TorchDataLoader

    train_loader = TorchDataLoader(
        train_set,
        batch_size=per_rank_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nb_workers,
        collate_fn=_collate_as_list,
        prefetch_factor=1,
    )
    val_loader = TorchDataLoader(
        val_set,
        batch_size=eval_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nb_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=1,
    )
    test_loader = TorchDataLoader(
        test_set,
        batch_size=eval_bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=nb_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=1,
    )

    base_model = (
        GCN(in_channels=in_channels, dropout=dropout_val)
        if arch == "gcn"
        else GAT(in_channels=in_channels, dropout=dropout_val)
    ).to(device)
    if pretrained_weights:
        if not os.path.exists(pretrained_weights):
            raise FileNotFoundError(f"Pre-trained weights file not found: {pretrained_weights}")
        if is_main:
            print(f"Loading pre-trained weights from '{pretrained_weights}' (with automatic channel adaptation if 5D->4D)...")
        sd = torch.load(pretrained_weights, map_location=device, weights_only=True)
        infer_feat_idx = feature_indices
        if infer_feat_idx is None and in_channels == 4:
            # Infer selected 4 features from data_folder name if possible
            folder_str = os.path.basename(os.path.normpath(data_folder))
            tag_map = {
                "no_RA": [1, 2, 3, 4],
                "no_Dec": [0, 2, 3, 4],
                "no_Parallax": [0, 1, 3, 4],
                "no_pmRA": [0, 1, 2, 4],
                "no_pmDec": [0, 1, 2, 3],
            }
            for k_tag, f_list in tag_map.items():
                if k_tag in folder_str:
                    infer_feat_idx = f_list
                    break
        load_pretrained_with_channel_adaptation(base_model, sd, selected_features=infer_feat_idx)

    model = DDP(base_model) if is_ddp else base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=Lr, weight_decay=weight_decay_val)
    if pos_weight_val != 1.0:
        criterion_sum = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device), reduction="sum"
        )
    else:
        criterion_sum = torch.nn.BCEWithLogitsLoss(reduction="sum")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_core_fnr = 100.0
    best_epoch = 1
    epochs_without_improvement = 0
    start_epoch = 0
    last_model_path = best_model_path.replace("best_", "last_")
    if last_model_path == best_model_path:
        last_model_path = best_model_path.replace(".pth", "_last.pth")

    # Resume checkpoint path for automatic recovery if SLURM restarts or OOM fallback triggers
    resume_ckpt_path = best_model_path.replace(".pth", "_resume_state.pt")
    ckpt = None
    if getattr(args, "resume", True) and os.path.exists(resume_ckpt_path):
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            base_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Explicitly enforce current weight_decay on all parameter groups upon resume
            for pg in optimizer.param_groups:
                pg["weight_decay"] = weight_decay_val
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_val_acc = float(ckpt.get("best_val_acc", 0.0))
            best_val_core_fnr = float(ckpt.get("best_val_core_fnr", 100.0))
            best_epoch = int(ckpt.get("best_epoch", 1))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
            if os.path.exists(log_filename):
                parsed_hist = parse_log_file(log_filename)
                if parsed_hist["val_loss"]:
                    min_vl = min(parsed_hist["val_loss"])
                    best_idx = parsed_hist["val_loss"].index(min_vl)
                    best_val_loss = float(min_vl)
                    best_val_acc = float(parsed_hist["val_acc"][best_idx])
                    best_epoch = int(parsed_hist["epochs"][best_idx])
                    epochs_without_improvement = max(0, int(parsed_hist["epochs"][-1]) - best_epoch)
            if is_main:
                print(
                    f"[Resume] Resumed from epoch {start_epoch}/{epochs} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, "
                    f"best_val_acc={best_val_acc:.2f}%, no_improv={epochs_without_improvement}, "
                    f"weight_decay={weight_decay_val:.1e})"
                )
        except Exception as e:
            if is_main:
                print(f"[Resume] Could not load resume state ({e}); starting from epoch 0.")
            ckpt = None
            start_epoch = 0
    elif getattr(args, "resume", True) and os.path.exists(last_model_path) and os.path.exists(log_filename):
        try:
            parsed_hist = parse_log_file(log_filename)
            if parsed_hist["epochs"]:
                last_ep = int(parsed_hist["epochs"][-1])
                sd = torch.load(last_model_path, map_location=device, weights_only=True)
                base_model.load_state_dict(sd)
                start_epoch = last_ep
                if parsed_hist.get("lr"):
                    last_logged_lr = float(parsed_hist["lr"][-1])
                    for pg in optimizer.param_groups:
                        pg["lr"] = last_logged_lr
                if parsed_hist["val_loss"]:
                    min_vl = min(parsed_hist["val_loss"])
                    best_idx = parsed_hist["val_loss"].index(min_vl)
                    best_val_loss = float(min_vl)
                    best_val_acc = float(parsed_hist["val_acc"][best_idx])
                    best_epoch = int(parsed_hist["epochs"][best_idx])
                    epochs_without_improvement = max(0, last_ep - best_epoch)
                if is_main:
                    print(
                        f"[Resume] Resumed weights from '{last_model_path}' at epoch {start_epoch}/{epochs} "
                        f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, "
                        f"best_val_acc={best_val_acc:.2f}%, no_improv={epochs_without_improvement})"
                    )
        except Exception as e:
            if is_main:
                print(f"[Resume] Could not resume from last weights ({e}); starting from epoch 0.")
            ckpt = None
            start_epoch = 0

    # Initialize adaptive learning rate scheduler (aware of start_epoch on resumed runs)
    if start_epoch == 0:
        if scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=Lr,
                epochs=epochs,
                steps_per_epoch=max(1, len(train_loader)),
                pct_start=0.15,
                div_factor=5.0,
                final_div_factor=50.0,
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(5, epochs // 2), T_mult=1, eta_min=Lr * 0.02
            )
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        # Resumed run: ensure smooth continuation without resetting LR to warmup
        if scheduler_type == "onecycle":
            loaded_sched = False
            saved_sched_dict = ckpt.get("scheduler_state_dict", {}) if ckpt else {}
            # Only restore if the checkpoint genuinely came from OneCycleLR with consistent progress
            if "total_steps" in saved_sched_dict and saved_sched_dict.get("last_epoch", 0) >= (start_epoch - 1) * len(train_loader):
                try:
                    candidate = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=Lr,
                        epochs=epochs,
                        steps_per_epoch=max(1, len(train_loader)),
                        pct_start=0.15,
                        div_factor=5.0,
                        final_div_factor=50.0,
                    )
                    candidate.load_state_dict(saved_sched_dict)
                    scheduler = candidate
                    loaded_sched = True
                except Exception:
                    loaded_sched = False
            if not loaded_sched:
                current_lr = optimizer.param_groups[0]["lr"]
                remaining_epochs = max(1, epochs - start_epoch)
                remaining_steps = remaining_epochs * max(1, len(train_loader))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining_steps, eta_min=Lr * 0.02
                )
                if is_main:
                    print(
                        f"[Resume] OneCycleLR adaptation: smoothly cosine-annealing over remaining {remaining_steps} steps "
                        f"({remaining_epochs} epochs) from {current_lr:.2e} to {Lr * 0.02:.2e}."
                    )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(5, epochs // 2), T_mult=1, eta_min=Lr * 0.02
            )
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass

    if is_main and start_epoch == 0:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(
                f"=== START of training (split_mode={split_mode}, in_channels={in_channels}, "
                f"pos_weight={pos_weight_val}, world_size={world_size}, eff_bs={effective_bs}, "
                f"micro_bs={micro_bs}, dropout={dropout_val}, weight_decay={weight_decay_val:.1e}, "
                f"scheduler={scheduler_type}, patience={early_stop_patience}) ===\n"
            )
            if split_mode == "ic":
                f.write(f"Val ICs : {val_ics}\n")
                f.write(f"Test ICs: {test_ics}\n")
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | CoreFNR (CoreRec)\n")

    offload_dir = os.path.join(
        os.environ.get("SCRATCH", "/scratch/cozma"),
        "tmp_autograd_offload",
        f"job_{os.environ.get('SLURM_JOB_ID', 'local')}_rank_{rank}",
    )

    for epoch in range(start_epoch, epochs):
        cur_lr_check = optimizer.param_groups[0]["lr"]
        # Strict validation BCE loss early stopping (unconditional on LR reduction)
        if epochs_without_improvement >= early_stop_patience:
            if is_main:
                print(
                    f"[Strict Early Stopping] Validation BCE loss has not improved for {epochs_without_improvement} "
                    f"consecutive epochs (patience={early_stop_patience}, best={best_val_loss:.4f} at epoch {best_epoch}, "
                    f"LR={cur_lr_check:.1e}). Cutting training immediately to prevent overfitting!",
                    flush=True,
                )
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_nodes_train = 0.0
        running_train_loss = 0.0
        running_train_correct = 0.0

        for batch_idx, data_list in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            batch_total_nodes = float(sum(d.num_nodes for d in data_list))
            if batch_total_nodes <= 0:
                continue

            # Split `data_list` into safe micro-batches so peak RAM stays well below node limit
            cur_micro = micro_bs
            # Dynamic RAM check: if node memory usage > 65% of limit, halve micro-batch on the fly
            rss_now = get_process_and_cgroup_rss_gb()
            if rss_now > 0.65 * mem_limit_gb and cur_micro > 2:
                cur_micro = max(2, cur_micro // 2)

            chunks = [data_list[i : i + cur_micro] for i in range(0, len(data_list), cur_micro)]
            num_chunks = len(chunks)
            batch_loss_sum = 0.0

            for chunk_idx, sub_list in enumerate(chunks):
                sub_batch = Batch.from_data_list(sub_list).to(device)
                sync_ctx = (
                    model.no_sync()
                    if (is_ddp and chunk_idx < num_chunks - 1)
                    else torch.enable_grad()
                )
                with sync_ctx:
                    with DiskActivationOffloader(
                        offload_dir=offload_dir,
                        mem_threshold_fraction=0.70,
                        force_enable=False,
                    ):
                        out = model(sub_batch).view(-1)
                        loss_sum = criterion_sum(out, sub_batch.y)
                        scaled_loss = loss_sum / batch_total_nodes
                        scaled_loss.backward()

                with torch.no_grad():
                    s_loss = float(loss_sum.item())
                    batch_loss_sum += s_loss
                    running_train_loss += s_loss
                    total_nodes_train += float(sub_batch.num_nodes)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    running_train_correct += float((preds == sub_batch.y).sum().item())

                del sub_batch, out, loss_sum, scaled_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            if scheduler_type == "onecycle":
                try:
                    scheduler.step()
                except Exception:
                    pass
            elif scheduler_type == "cosine":
                scheduler.step(epoch + (batch_idx + 1) / max(1, len(train_loader)))

            if is_main and ((batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader)):
                rss_gb = get_process_and_cgroup_rss_gb()
                cur_step_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:03d} | Batch {batch_idx+1:04d}/{len(train_loader)} | "
                    f"Batch Loss: {batch_loss_sum / batch_total_nodes:.4f} | LR: {cur_step_lr:.2e} | RAM: {rss_gb:.1f}G/{mem_limit_gb:.0f}G",
                    flush=True,
                )

        gc.collect()

        model.eval()
        total_nodes_val = 0.0
        running_val_loss = 0.0
        running_val_correct = 0.0
        val_core_tp = 0.0
        val_core_fn = 0.0

        with torch.no_grad():
            for sub_batch in val_loader:
                sub_batch = sub_batch.to(device)
                out = base_model(sub_batch).view(-1)
                loss_sum = criterion_sum(out, sub_batch.y)

                running_val_loss += float(loss_sum.item())
                total_nodes_val += float(sub_batch.num_nodes)

                preds = (torch.sigmoid(out) > 0.5).float()
                running_val_correct += float((preds == sub_batch.y).sum().item())
                val_core_tp += float(((preds == 1) & (sub_batch.y == 1)).sum().item())
                val_core_fn += float(((preds == 0) & (sub_batch.y == 1)).sum().item())

        if is_ddp:
            metrics_t = torch.tensor(
                [
                    running_train_loss,
                    running_train_correct,
                    total_nodes_train,
                    running_val_loss,
                    running_val_correct,
                    total_nodes_val,
                    val_core_tp,
                    val_core_fn,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
            (
                running_train_loss,
                running_train_correct,
                total_nodes_train,
                running_val_loss,
                running_val_correct,
                total_nodes_val,
                val_core_tp,
                val_core_fn,
            ) = [float(x) for x in metrics_t.tolist()]

        avg_train_loss = running_train_loss / max(1.0, total_nodes_train)
        avg_val_loss = running_val_loss / max(1.0, total_nodes_val)
        train_acc = (running_train_correct / max(1.0, total_nodes_train)) * 100.0
        val_acc = (running_val_correct / max(1.0, total_nodes_val)) * 100.0
        val_core_total = max(1.0, val_core_tp + val_core_fn)
        val_core_fnr = (val_core_fn / val_core_total) * 100.0
        val_core_rec = (val_core_tp / val_core_total) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler_type == "plateau":
            scheduler.step(avg_val_loss)

        improved = avg_val_loss < (best_val_loss - 1e-4)
        if improved:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_core_fnr = val_core_fnr
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_main:
            star_str = " (*)" if (improved or not os.path.exists(best_model_path)) else ""
            log_line = (
                f"{epoch+1:03d}    | {avg_train_loss:.4f}     | {train_acc:.2f}%   | "
                f"{avg_val_loss:.4f}   | {val_acc:.2f}%  | {current_lr:.1e}{star_str} | "
                f"CoreFNR: {val_core_fnr:.2f}% (CoreRec: {val_core_rec:.2f}%)"
            )
            print("-" * 96)
            print(
                f"FIN EPOCH {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | "
                f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%) | Core FNR: {val_core_fnr:.2f}% (Rec: {val_core_rec:.2f}%) | LR: {current_lr:.1e}",
                flush=True,
            )
            print("-" * 96)

            _atomic_save_state_dict(base_model.state_dict(), last_model_path)

            if improved or not os.path.exists(best_model_path):
                _atomic_save_state_dict(base_model.state_dict(), best_model_path)
                _archive_and_register_weights(
                    best_model_path,
                    arch,
                    in_channels,
                    best_epoch,
                    best_val_loss,
                    best_val_acc,
                    val_core_fnr=best_val_core_fnr,
                )
                print(
                    f"      --> Best model saved to '{best_model_path}' and mirrored to '{CENTRAL_WEIGHTS_DIR}'!",
                    flush=True,
                )

            # Save atomic resume state after every epoch
            tmp_resume = resume_ckpt_path + ".tmp"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_val_core_fnr": best_val_core_fnr,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                tmp_resume,
            )
            os.replace(tmp_resume, resume_ckpt_path)

            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

            try:
                update_live_plots_for_log(log_filename)
                subprocess.Popen(
                    [sys.executable, os.path.join(ROOT_DIR, "scripts", "live_monitor.py"), "--once"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        if is_ddp:
            dist.barrier()

    if is_main:
        if not os.path.exists(best_model_path):
            _atomic_save_state_dict(base_model.state_dict(), best_model_path)
        print("\nFinal test evaluation...", flush=True)
    if os.path.exists(best_model_path):
        base_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    base_model.eval()

    total_nodes_test = 0.0
    running_test_loss = 0.0
    running_test_correct = 0.0
    tp_te = fp_te = tn_te = fn_te = 0.0

    with torch.no_grad():
        for sub_batch in test_loader:
            sub_batch = sub_batch.to(device)
            out = base_model(sub_batch).view(-1)
            loss_sum = criterion_sum(out, sub_batch.y)

            running_test_loss += float(loss_sum.item())
            total_nodes_test += float(sub_batch.num_nodes)

            preds = (torch.sigmoid(out) > 0.5).float()
            running_test_correct += float((preds == sub_batch.y).sum().item())
            tp_te += float(((preds == 1) & (sub_batch.y == 1)).sum().item())
            fp_te += float(((preds == 1) & (sub_batch.y == 0)).sum().item())
            tn_te += float(((preds == 0) & (sub_batch.y == 0)).sum().item())
            fn_te += float(((preds == 0) & (sub_batch.y == 1)).sum().item())

    if is_ddp:
        test_t = torch.tensor(
            [running_test_loss, running_test_correct, total_nodes_test, tp_te, fp_te, tn_te, fn_te],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(test_t, op=dist.ReduceOp.SUM)
        running_test_loss, running_test_correct, total_nodes_test, tp_te, fp_te, tn_te, fn_te = [
            float(x) for x in test_t.tolist()
        ]

    if is_main:
        avg_test_loss = running_test_loss / max(1.0, total_nodes_test)
        test_acc = (running_test_correct / max(1.0, total_nodes_test)) * 100.0
        test_prec = tp_te / max(1.0, tp_te + fp_te)
        test_rec = tp_te / max(1.0, tp_te + fn_te)
        test_spec = tn_te / max(1.0, tn_te + fp_te)
        test_f1 = 2.0 * test_prec * test_rec / max(1e-12, test_prec + test_rec)
        test_bal_acc = 50.0 * (test_rec + test_spec)
        test_core_fnr = (fn_te / max(1.0, tp_te + fn_te)) * 100.0
        test_core_rec = test_rec * 100.0

        _archive_and_register_weights(
            best_model_path,
            arch,
            in_channels,
            best_epoch,
            best_val_loss,
            best_val_acc,
            test_loss=avg_test_loss,
            test_acc=test_acc,
            test_f1=test_f1,
            val_core_fnr=best_val_core_fnr,
            test_core_fnr=test_core_fnr,
        )

        print(
            f"--> Final Test Loss: {avg_test_loss:.4f} | Final Test Acc: {test_acc:.2f}% | "
            f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} (Prec: {test_prec:.4f}, Rec: {test_rec:.4f}) | "
            f"CoreFNR: {test_core_fnr:.2f}% (CoreRec: {test_core_rec:.2f}%)",
            flush=True,
        )

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write("-" * 50 + "\n")
            f.write(
                f"Final Test Loss : {avg_test_loss:.4f} | Acc: {test_acc:.2f}% | "
                f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | "
                f"CoreFNR: {test_core_fnr:.2f}% | CoreRec: {test_core_rec:.2f}%\n"
            )
            f.write("=== END OF TRAINING ===\n")

        if os.path.exists(resume_ckpt_path):
            try:
                os.remove(resume_ckpt_path)
            except OSError:
                pass

        try:
            update_live_plots_for_log(log_filename)
            subprocess.run(
                [sys.executable, os.path.join(ROOT_DIR, "scripts", "live_monitor.py"), "--once"],
                check=False,
                timeout=30,
            )
        except Exception:
            pass

        print(
            f"Training finished. Logs written to '{log_filename}'. "
            f"Best model saved at '{best_model_path}' and mirrored in '{CENTRAL_WEIGHTS_DIR}'.",
            flush=True,
        )

    if is_ddp:
        dist.destroy_process_group()
