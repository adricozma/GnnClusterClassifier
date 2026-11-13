"""Distributed Study 2 V2 trainer for the 3-layer Multi-Head Residual GAT_V2 classifier.

Supports:
- Simulation-level IC partitioning (zero leakage, 80/10/10 seed=42 matching V1).
- 3-class taxonomy (`0: Unbound/Field`, `1: Potential Escaper`, `2: Bound Core`) with
  class-balanced CrossEntropyLoss or binary BCE fallback (`--num_classes 1`).
- Feature-index slicing + uncertainty-channel leakage masking for Best-4D runs.
- Variance-preserving `5D -> 4D` / `Clean -> Noisy` weight initialization via `--pretrained_weights`.
- Multi-node CPU DDP (`torchrun` / `srun`), gradient accumulation (`--bs` / `--micro_bs`),
  `DiskActivationOffloader`, `OneCycleLR` / `CosineAnnealingLR`, and atomic checkpoint resume.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch, Data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_models_v2 import (
    CLASS_NAMES_3WAY,
    GAT_V2,
    load_pretrained_with_channel_adaptation_v2,
)
from oom_Pin_and_offload import DiskActivationOffloader, init_distributed


SNAP_ROT_RE = re.compile(r"^graph_v2_(.+)_rot_(\d+)\.pt$")


def parse_feature_indices(raw: str) -> List[int]:
    if not raw.strip():
        return [0, 1, 2, 3, 4]
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def mask_err_tensor(err_x: Optional[torch.Tensor], feat_indices: Sequence[int]) -> Optional[torch.Tensor]:
    """Zero out uncertainty channels corresponding to dropped astrometric dimensions."""
    if err_x is None:
        return None
    kept = set(int(i) for i in feat_indices)
    if kept.issuperset({2, 3, 4}):
        return err_x
    out = err_x.clone()
    if 2 not in kept:
        out[:, 0] = 0.0
        out[:, 3] = 0.0
    if 3 not in kept:
        out[:, 1] = 0.0
    if 4 not in kept:
        out[:, 2] = 0.0
    return out


def discover_ic_splits(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Deterministic simulation-level IC split matching Study 2 V1 (`seed=42`)."""
    ic_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if d.startswith("IC_") and os.path.isdir(os.path.join(data_dir, d))
        ],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )
    if len(ic_dirs) == 0:
        raise RuntimeError(f"No IC_* directories found in '{data_dir}'.")
    if len(ic_dirs) < 3:
        return ic_dirs, ic_dirs, ic_dirs

    n_total = len(ic_dirs)
    n_train = max(1, int(round(0.8 * n_total)))
    n_val = max(1, int(round(0.1 * n_total)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_test = n_total - n_train - n_val

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_ics = [ic_dirs[i] for i in perm[:n_train]]
    val_ics = [ic_dirs[i] for i in perm[n_train : n_train + n_val]]
    test_ics = [ic_dirs[i] for i in perm[n_train + n_val : n_train + n_val + n_test]]
    return train_ics, val_ics, test_ics


def index_split_graphs(
    data_dir: str,
    ic_names: Sequence[str],
) -> Dict[int, List[str]]:
    """Map rotation index -> sorted list of graph `.pt` paths for an IC subset."""
    by_rot: Dict[int, List[str]] = {}
    for ic in ic_names:
        ic_path = os.path.join(data_dir, ic)
        if not os.path.isdir(ic_path):
            continue
        for fname in sorted(os.listdir(ic_path)):
            m = SNAP_ROT_RE.match(fname)
            if not m:
                continue
            rot = int(m.group(2))
            by_rot.setdefault(rot, []).append(os.path.join(ic_path, fname))
    return by_rot


def load_graph_item(
    path: str,
    feat_indices: Sequence[int],
    num_classes: int,
) -> Data:
    """Load one PyG graph and apply feature slicing + uncertainty leakage masking."""
    data: Data = torch.load(path, map_location="cpu", weights_only=False)
    if data.x.shape[1] > len(feat_indices):
        data.x = data.x[:, feat_indices].contiguous()
    if hasattr(data, "err_x") and data.err_x is not None:
        data.err_x = mask_err_tensor(data.err_x, feat_indices)
    if num_classes == 1:
        data.y = (data.y == 2).to(torch.float32)
    else:
        data.y = data.y.to(torch.long)
    return data


def compute_empirical_class_weights(
    paths: Sequence[str],
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute smoothed inverse-frequency weights w_c proportional to (N / N_c)^power."""
    counts = np.zeros(num_classes, dtype=np.float64)
    sample_paths = paths[: min(len(paths), 200)]
    for p in sample_paths:
        g = torch.load(p, map_location="cpu", weights_only=False)
        y = g.y.numpy()
        for c in range(num_classes):
            counts[c] += float(np.sum(y == c))
    counts = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    raw_w = (total / counts) ** float(power)
    raw_w = raw_w / np.mean(raw_w)
    return torch.tensor(raw_w, dtype=torch.float32)


def compute_confusion_metrics(
    conf_mat: np.ndarray,
    field_fp: int,
    field_total: int,
    num_classes: int,
) -> Dict[str, float]:
    """Derive balanced accuracy, macro F1, Bound F1/Precision/Recall, CoreFNR, and FieldFPR."""
    total = max(float(np.sum(conf_mat)), 1.0)
    acc = 100.0 * float(np.trace(conf_mat)) / total

    recalls: List[float] = []
    f1s: List[float] = []
    precs: List[float] = []
    for c in range(num_classes):
        tp = float(conf_mat[c, c])
        fn = float(np.sum(conf_mat[c, :]) - tp)
        fp = float(np.sum(conf_mat[:, c]) - tp)
        rec = tp / max(tp + fn, 1.0)
        prec = tp / max(tp + fp, 1.0)
        f1 = (2.0 * prec * rec) / max(prec + rec, 1e-8)
        recalls.append(rec)
        precs.append(prec)
        f1s.append(f1)

    bound_idx = num_classes - 1
    bound_rec = recalls[bound_idx] * 100.0
    bound_prec = precs[bound_idx]
    bound_f1 = f1s[bound_idx]
    core_fnr = 100.0 - bound_rec

    pe_rec = recalls[1] * 100.0 if num_classes == 3 else 0.0
    field_fpr = 100.0 * float(field_fp) / max(float(field_total), 1.0)

    return {
        "acc": acc,
        "bal_acc": 100.0 * float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "bound_f1": bound_f1,
        "bound_prec": bound_prec,
        "bound_rec": bound_rec,
        "core_fnr": core_fnr,
        "pe_rec": pe_rec,
        "field_fpr": field_fpr,
    }


def evaluate_split(
    model: nn.Module,
    paths: Sequence[str],
    feat_indices: Sequence[int],
    num_classes: int,
    criterion: nn.Module,
    micro_bs: int,
    is_distributed: bool,
    rank: int,
    world_size: int,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Distributed evaluation over an evaluation split."""
    model.eval()
    shard = list(paths[rank::world_size])
    n_c = 2 if num_classes == 1 else num_classes
    conf = np.zeros((n_c, n_c), dtype=np.int64)
    loss_sum = 0.0
    node_count = 0
    field_fp = 0
    field_total = 0

    with torch.no_grad():
        for i in range(0, len(shard), micro_bs):
            chunk = [load_graph_item(p, feat_indices, num_classes) for p in shard[i : i + micro_bs]]
            if not chunk:
                continue
            batch = Batch.from_data_list(chunk)
            logits = model(
                batch.x,
                batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                err_x=getattr(batch, "err_x", None),
                batch=batch.batch,
                ptr=getattr(batch, "ptr", None),
            )
            if num_classes == 1:
                loss = criterion(logits.squeeze(-1), batch.y)
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs >= 0.5).to(torch.long)
                targets = batch.y.to(torch.long)
            else:
                loss = criterion(logits, batch.y)
                preds = torch.argmax(logits, dim=-1)
                targets = batch.y

            n_nodes = int(batch.y.numel())
            loss_sum += float(loss.item()) * n_nodes
            node_count += n_nodes

            p_np = preds.cpu().numpy()
            t_np = targets.cpu().numpy()
            for t_val, p_val in zip(t_np, p_np):
                conf[t_val, p_val] += 1

            if hasattr(batch, "is_field") and batch.is_field is not None:
                f_mask = batch.is_field.cpu().numpy().astype(bool)
                field_total += int(np.sum(f_mask))
                field_fp += int(np.sum(p_np[f_mask] == (n_c - 1)))

    if is_distributed:
        vec = torch.tensor(
            [loss_sum, float(node_count), float(field_fp), float(field_total)]
            + conf.astype(np.float64).ravel().tolist(),
            dtype=torch.float64,
        )
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        loss_sum = float(vec[0].item())
        node_count = int(vec[1].item())
        field_fp = int(vec[2].item())
        field_total = int(vec[3].item())
        conf = vec[4:].numpy().reshape(n_c, n_c).astype(np.int64)

    avg_loss = loss_sum / max(node_count, 1)
    metrics = compute_confusion_metrics(conf, field_fp, field_total, n_c)
    return avg_loss, metrics, conf


def atomic_torch_save(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Study 2 V2 GAT_V2 classifier.")
    p.add_argument("--data_folder", type=str, required=True)
    p.add_argument("--n_rotations", type=int, default=5)
    p.add_argument("--feature_indices", type=str, default="0,1,2,3,4")
    p.add_argument("--num_classes", type=int, default=3, choices=[1, 3])
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--edge_dim", type=int, default=3)
    p.add_argument("--err_features", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--micro_bs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--pretrained_weights", type=str, default="")
    p.add_argument("--save_model", type=str, required=True)
    p.add_argument("--log_file", type=str, required=True)
    p.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "cosine"])
    p.add_argument("--stochastic_rotation", action="store_true", default=True)
    p.add_argument("--no_stochastic_rotation", dest="stochastic_rotation", action="store_false")
    p.add_argument("--force_disk_offload", action="store_true")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    is_distributed, rank, _, world_size = init_distributed()
    is_master = rank == 0

    feat_indices = parse_feature_indices(args.feature_indices)
    in_features = len(feat_indices)

    train_ics, val_ics, test_ics = discover_ic_splits(args.data_folder)
    train_by_rot = index_split_graphs(args.data_folder, train_ics)
    val_by_rot = index_split_graphs(args.data_folder, val_ics)
    test_by_rot = index_split_graphs(args.data_folder, test_ics)

    available_rots = sorted(train_by_rot.keys())
    if not available_rots:
        raise RuntimeError(f"No graph_v2_*_rot_*.pt files found inside '{args.data_folder}'.")

    val_paths = val_by_rot.get(available_rots[0], sum(val_by_rot.values(), []))
    test_paths = test_by_rot.get(available_rots[0], sum(test_by_rot.values(), []))

    model = GAT_V2(
        in_features=in_features,
        err_features=args.err_features,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        out_channels=args.num_classes,
        edge_dim=args.edge_dim,
        dropout=args.dropout,
    )

    if args.num_classes == 1:
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    else:
        if is_master:
            cw = compute_empirical_class_weights(
                train_by_rot[available_rots[0]], args.num_classes, power=args.class_weight_power
            )
        else:
            cw = torch.ones(args.num_classes, dtype=torch.float32)
        if is_distributed:
            dist.broadcast(cw, src=0)
        criterion = nn.CrossEntropyLoss(weight=cw)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(args.save_model)), "latest_v2_ckpt.pth")
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        patience_counter = int(ckpt.get("patience_counter", 0))
        if is_master:
            print(f"[Resume] Restored checkpoint at epoch {start_epoch} (best_val={best_val_loss:.4f}).")
    elif args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        state = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        loaded, skipped = load_pretrained_with_channel_adaptation_v2(
            model, state, selected_features=feat_indices, d_err=args.err_features
        )
        if is_master:
            print(
                f"[Transfer] Loaded {loaded} tensors (skipped {skipped}) from '{args.pretrained_weights}'."
            )

    train_model: nn.Module = DDP(model) if is_distributed else model
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    sample_train_len = (
        len(train_by_rot[available_rots[0]])
        if args.stochastic_rotation
        else sum(len(v) for v in train_by_rot.values())
    )
    steps_per_epoch = max(1, math.ceil(sample_train_len / max(args.bs, 1)))
    total_sched_steps = max(args.epochs * steps_per_epoch, 1)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_sched_steps,
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_sched_steps
        )
    sched_steps_done = 0

    if args.resume and os.path.isfile(ckpt_path) and start_epoch > 0:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        sched_steps_done = int(ckpt.get("sched_steps_done", 0))

    offloader = DiskActivationOffloader(force_enable=args.force_disk_offload)

    if is_master:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
        mode_str = "a" if (args.resume and start_epoch > 0 and os.path.isfile(args.log_file)) else "w"
        with open(args.log_file, mode_str, encoding="utf-8") as lf:
            if mode_str == "w":
                lf.write(
                    f"Study 2 V2 GAT_V2 | features={feat_indices} | num_classes={args.num_classes} | "
                    f"IC split: {len(train_ics)}/{len(val_ics)}/{len(test_ics)}\n"
                )
                lf.write(
                    "Epoch | TrLoss | TrAcc   | VaLoss | VaAcc   | BoundF1 | PE_Rec  | CoreFNR | FieldFPR | LR\n"
                )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_model.train()

        if args.stochastic_rotation:
            rot_idx = available_rots[epoch % len(available_rots)]
            epoch_paths = list(train_by_rot[rot_idx])
        else:
            epoch_paths = [p for r in available_rots for p in train_by_rot[r]]

        rng = np.random.default_rng(10000 + epoch)
        perm = rng.permutation(len(epoch_paths))
        epoch_paths = [epoch_paths[i] for i in perm]

        # Ensure equal step counts across DDP ranks
        n_batches = max(1, len(epoch_paths) // max(args.bs, 1))
        if n_batches * args.bs < len(epoch_paths) and len(epoch_paths) < args.bs:
            n_batches = 1

        tr_loss_sum = 0.0
        tr_correct = 0
        tr_nodes = 0

        for b_idx in range(n_batches):
            global_batch = epoch_paths[b_idx * args.bs : (b_idx + 1) * args.bs]
            if not global_batch:
                global_batch = epoch_paths
            rank_paths = global_batch[rank::world_size]

            optimizer.zero_grad(set_to_none=True)
            micro_chunks = [
                rank_paths[j : j + args.micro_bs]
                for j in range(0, len(rank_paths), args.micro_bs)
            ] or [[]]

            n_micro = len(micro_chunks)
            for m_idx, chunk_paths in enumerate(micro_chunks):
                if not chunk_paths:
                    continue
                graphs = [load_graph_item(p, feat_indices, args.num_classes) for p in chunk_paths]
                batch = Batch.from_data_list(graphs)

                sync_ctx = (
                    train_model.no_sync()
                    if (is_distributed and m_idx < n_micro - 1)
                    else nullcontext()
                )
                with sync_ctx, offloader.offload_context():
                    logits = train_model(
                        batch.x,
                        batch.edge_index,
                        edge_attr=getattr(batch, "edge_attr", None),
                        err_x=getattr(batch, "err_x", None),
                        batch=batch.batch,
                        ptr=getattr(batch, "ptr", None),
                    )
                    if args.num_classes == 1:
                        loss = criterion(logits.squeeze(-1), batch.y)
                        preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).to(torch.long)
                        targets = batch.y.to(torch.long)
                    else:
                        loss = criterion(logits, batch.y)
                        preds = torch.argmax(logits, dim=-1)
                        targets = batch.y
                    (loss / float(max(n_micro, 1))).backward()

                n_n = int(batch.y.numel())
                tr_loss_sum += float(loss.item()) * n_n
                tr_correct += int((preds == targets).sum().item())
                tr_nodes += n_n

            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=2.0)
            optimizer.step()
            if sched_steps_done < total_sched_steps:
                scheduler.step()
                sched_steps_done += 1

        if is_distributed:
            tr_vec = torch.tensor([tr_loss_sum, float(tr_correct), float(tr_nodes)], dtype=torch.float64)
            dist.all_reduce(tr_vec, op=dist.ReduceOp.SUM)
            tr_loss = float(tr_vec[0].item() / max(tr_vec[2].item(), 1.0))
            tr_acc = 100.0 * float(tr_vec[1].item() / max(tr_vec[2].item(), 1.0))
        else:
            tr_loss = tr_loss_sum / max(tr_nodes, 1)
            tr_acc = 100.0 * float(tr_correct) / max(tr_nodes, 1)

        va_loss, va_m, _ = evaluate_split(
            train_model,
            val_paths,
            feat_indices,
            args.num_classes,
            criterion,
            args.micro_bs,
            is_distributed,
            rank,
            world_size,
        )

        improved = va_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = va_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if is_master:
            raw_state = (train_model.module if is_distributed else train_model).state_dict()
            if improved:
                atomic_torch_save({"model_state": raw_state, "epoch": epoch, "val_loss": va_loss}, args.save_model)
            atomic_torch_save(
                {
                    "model_state": raw_state,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "sched_steps_done": sched_steps_done,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                },
                ckpt_path,
            )
            lr_now = optimizer.param_groups[0]["lr"]
            star = " *" if improved else ""
            line = (
                f"{epoch + 1:03d} | {tr_loss:.4f} | {tr_acc:6.2f}% | {va_loss:.4f} | {va_m['acc']:6.2f}% | "
                f"{va_m['bound_f1']:.4f} | {va_m['pe_rec']:6.2f}% | {va_m['core_fnr']:6.2f}% | "
                f"{va_m['field_fpr']:6.2f}% | {lr_now:.1e} ({time.time() - t0:.1f}s){star}\n"
            )
            print(line, end="")
            with open(args.log_file, "a", encoding="utf-8") as lf:
                lf.write(line)

        if patience_counter >= args.patience:
            if is_master:
                print(f"[EarlyStopping] Stopped at epoch {epoch + 1} (patience={args.patience}).")
            break

    if is_distributed:
        dist.barrier()

    # Final evaluation on held-out Test ICs using best validation checkpoint
    if os.path.isfile(args.save_model):
        best_ckpt = torch.load(args.save_model, map_location="cpu", weights_only=False)
        target_mod = train_model.module if is_distributed else train_model
        target_mod.load_state_dict(best_ckpt["model_state"])

    te_loss, te_m, te_conf = evaluate_split(
        train_model,
        test_paths,
        feat_indices,
        args.num_classes,
        criterion,
        args.micro_bs,
        is_distributed,
        rank,
        world_size,
    )

    if is_master:
        summary_line = (
            f"Final Test Loss : {te_loss:.4f} | Acc: {te_m['acc']:.2f}% | "
            f"BalAcc: {te_m['bal_acc']:.2f}% | MacroF1: {te_m['macro_f1']:.4f} | "
            f"F1: {te_m['bound_f1']:.4f} | Prec: {te_m['bound_prec']:.4f} | "
            f"Rec: {te_m['bound_rec']:.4f} | PE_Rec: {te_m['pe_rec']:.2f}% | "
            f"CoreFNR: {te_m['core_fnr']:.2f}% | FieldFPR: {te_m['field_fpr']:.2f}%\n"
        )
        print("\n" + summary_line)
        print("Test Confusion Matrix (rows=True, cols=Pred):", CLASS_NAMES_3WAY if args.num_classes == 3 else ("Unbound", "Bound"))
        print(te_conf)
        with open(args.log_file, "a", encoding="utf-8") as lf:
            lf.write("\n" + summary_line)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
