""" Unified distributed / gradient-accumulation / disk-offload training engine for GCN and GAT."""

import gc
import math
import os
import subprocess
import sys
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

from gnn_models import GAT, GCN, load_pretrained_with_channel_adaptation
from live_plot_utils import update_live_plots_for_log
from oom_Pin_and_offload import (
    DiskActivationOffloader,
    get_memory_limit_gb,
    get_process_and_cgroup_rss_gb,
    init_distributed,
)


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

    default_micro = 32 if arch == "gcn" else 8
    micro_bs = getattr(args, "micro_bs", 0) or default_micro

    if is_main:
        os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)

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
    micro_bs = min(micro_bs, per_rank_bs)
    mem_limit_gb = get_memory_limit_gb(246.0)

    if is_main:
        print(
            f"[{arch.upper()}] Device: {device} | DDP World Size: {world_size} node(s) | "
            f"Threads/rank: {torch.get_num_threads()} | RAM Limit/node: {mem_limit_gb:.1f} GB"
        )
        print(
            f"[{arch.upper()}] Effective Global Batch Size: {effective_bs} -> "
            f"{per_rank_bs}/node (Micro-batch size: {micro_bs} graphs, "
            f"{math.ceil(per_rank_bs / micro_bs)} grad-accum steps/node)"
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
        batch_size=micro_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nb_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=1,
    )
    test_loader = TorchDataLoader(
        test_set,
        batch_size=micro_bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=nb_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=1,
    )

    base_model = (GCN(in_channels=in_channels) if arch == "gcn" else GAT(in_channels=in_channels)).to(device)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=Lr, weight_decay=1e-4)
    if pos_weight_val != 1.0:
        criterion_sum = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device), reduction="sum"
        )
    else:
        criterion_sum = torch.nn.BCEWithLogitsLoss(reduction="sum")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_val_loss = float("inf")
    start_epoch = 0

    # Resume checkpoint path for automatic recovery if SLURM restarts or OOM fallback triggers
    resume_ckpt_path = best_model_path.replace(".pth", "_resume_state.pt")
    if getattr(args, "resume", True) and os.path.exists(resume_ckpt_path):
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            base_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            if is_main:
                print(f"[Resume] Resumed from epoch {start_epoch}/{epochs} (best_val_loss={best_val_loss:.4f})")
        except Exception as e:
            if is_main:
                print(f"[Resume] Could not load resume state ({e}); starting from epoch 0.")

    if is_main and start_epoch == 0:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(
                f"=== START of training (split_mode={split_mode}, in_channels={in_channels}, "
                f"pos_weight={pos_weight_val}, world_size={world_size}, eff_bs={effective_bs}, micro_bs={micro_bs}) ===\n"
            )
            if split_mode == "ic":
                f.write(f"Val ICs : {val_ics}\n")
                f.write(f"Test ICs: {test_ics}\n")
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR\n")

    offload_dir = os.path.join(
        os.environ.get("SCRATCH", "/scratch/cozma"),
        "tmp_autograd_offload",
        f"job_{os.environ.get('SLURM_JOB_ID', 'local')}_rank_{rank}",
    )

    for epoch in range(start_epoch, epochs):
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
                        force_enable=getattr(args, "force_disk_offload", False),
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

            if is_main and ((batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader)):
                rss_gb = get_process_and_cgroup_rss_gb()
                print(
                    f"Epoch {epoch+1:03d} | Batch {batch_idx+1:04d}/{len(train_loader)} | "
                    f"Batch Loss: {batch_loss_sum / batch_total_nodes:.4f} | RAM: {rss_gb:.1f}G/{mem_limit_gb:.0f}G",
                    flush=True,
                )

        gc.collect()

        model.eval()
        total_nodes_val = 0.0
        running_val_loss = 0.0
        running_val_correct = 0.0

        with torch.no_grad():
            for sub_batch in val_loader:
                sub_batch = sub_batch.to(device)
                out = base_model(sub_batch).view(-1)
                loss_sum = criterion_sum(out, sub_batch.y)

                running_val_loss += float(loss_sum.item())
                total_nodes_val += float(sub_batch.num_nodes)

                preds = (torch.sigmoid(out) > 0.5).float()
                running_val_correct += float((preds == sub_batch.y).sum().item())

        if is_ddp:
            metrics_t = torch.tensor(
                [
                    running_train_loss,
                    running_train_correct,
                    total_nodes_train,
                    running_val_loss,
                    running_val_correct,
                    total_nodes_val,
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
            ) = [float(x) for x in metrics_t.tolist()]

        avg_train_loss = running_train_loss / max(1.0, total_nodes_train)
        avg_val_loss = running_val_loss / max(1.0, total_nodes_val)
        train_acc = (running_train_correct / max(1.0, total_nodes_train)) * 100.0
        val_acc = (running_val_correct / max(1.0, total_nodes_val)) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(avg_val_loss)

        if is_main:
            log_line = (
                f"{epoch+1:03d}    | {avg_train_loss:.4f}     | {train_acc:.2f}%   | "
                f"{avg_val_loss:.4f}   | {val_acc:.2f}%  | {current_lr:.1e}"
            )
            print("-" * 80)
            print(
                f"FIN EPOCH {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | "
                f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%) | LR: {current_lr:.1e}",
                flush=True,
            )
            print("-" * 80)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(base_model.state_dict(), best_model_path)
                print("      --> Best model saved!", flush=True)
                log_line += " (*)"

            # Save atomic resume state after every epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                resume_ckpt_path,
            )

            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

            try:
                update_live_plots_for_log(log_filename)
                if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
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

        print(
            f"--> Final Test Loss: {avg_test_loss:.4f} | Final Test Acc: {test_acc:.2f}% | "
            f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} (Prec: {test_prec:.4f}, Rec: {test_rec:.4f})",
            flush=True,
        )

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write("-" * 50 + "\n")
            f.write(
                f"Final Test Loss : {avg_test_loss:.4f} | Acc: {test_acc:.2f}% | "
                f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f}\n"
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

        print(f"Training finished. Logs written to '{log_filename}'. Best model is '{best_model_path}'.", flush=True)

    if is_ddp:
        dist.destroy_process_group()
