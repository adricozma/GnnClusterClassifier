"""Training and curriculum fine-tuning script for the 3-layer GCN classifier (GCN-01).

Supports:
- Simulation-level (`--split_mode ic`) train/val/test partitioning to prevent data leakage
  across snapshots and 3D rotations of the same progenitor cluster.
- Two-stage Curriculum Learning:
  Stage 1: Pre-training on noiseless, unbiased 5D astrometric graphs.
  Stage 2: Fine-tuning (`--pretrained_weights`) on Gaia DR3 noise-injected graphs.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GCN(torch.nn.Module):
    """Three-layer Graph Convolutional Network for node-level bound/unbound classification."""

    def __init__(self, in_channels=5, hidden1=32, hidden2=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class LazyGraphDataset(Dataset):
    """On-demand PyTorch Geometric dataset loading `.pt` graphs from disk."""

    def __init__(self, file_pairs):
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        data_path, label_path = self.file_pairs[idx]
        data = torch.load(data_path, weights_only=False)
        target = torch.load(label_path, weights_only=False)
        data.y = target.float()
        return data


def split_by_simulation(file_pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Group all snapshots and rotations of each `IC_<id>` into the same split."""
    ic_to_pairs = {}
    for pair in file_pairs:
        ic_dir = os.path.basename(os.path.dirname(pair[0]))
        ic_to_pairs.setdefault(ic_dir, []).append(pair)

    ic_names = sorted(ic_to_pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(ic_names)

    n_ics = len(ic_names)
    n_train = int(round(train_ratio * n_ics))
    n_val = max(1, int(round(val_ratio * n_ics)))
    if n_train + n_val >= n_ics and n_ics >= 3:
        n_train = max(1, n_ics - 2)
        n_val = 1

    train_ics = set(ic_names[:n_train])
    val_ics = set(ic_names[n_train:n_train + n_val])

    train_pairs, val_pairs, test_pairs = [], [], []
    for ic_name, pairs in ic_to_pairs.items():
        if ic_name in train_ics:
            train_pairs.extend(pairs)
        elif ic_name in val_ics:
            val_pairs.extend(pairs)
        else:
            test_pairs.extend(pairs)

    return train_pairs, val_pairs, test_pairs


def compute_binary_metrics(tp, fp, tn, fn):
    """Compute Accuracy, Precision, Recall, F1, Balanced Accuracy, and MCC."""
    total = max(1, tp + fp + tn + fn)
    acc = 100.0 * (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    bal_acc = 50.0 * (recall + specificity)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return acc, precision, recall, f1, bal_acc, mcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or fine-tune the 3-layer GCN model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get(
            "GRAPH_DATA_DIR",
            os.path.join(ROOT_DIR, "GNN-jed", "graph_data") + "/",
        ),
        help="Path to graph_data directory.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="ic",
        choices=["ic", "random"],
        help="Partition dataset by progenitor simulation ('ic') or randomly ('random').",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight.")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Optional path to pre-trained .pth weights for Stage 2 curriculum fine-tuning.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="best_gcn_model.pth",
        help="Output path for best model checkpoint.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="training_metrics_gcn.log",
        help="Output path for training log.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = args.data_dir
    outputlist = list(range(0, 101))
    n_rotations = 5

    file_pairs = []
    for d in sorted(os.listdir(data_folder)):
        if d.startswith("IC_"):
            data_folder_path = os.path.join(data_folder, d)
            if os.path.isdir(data_folder_path):
                for output in outputlist:
                    for i in range(n_rotations):
                        g_path = os.path.join(
                            data_folder_path, f"graph_data_output_{output}_rotation_{i}.pt"
                        )
                        l_path = os.path.join(
                            data_folder_path, f"label_output_{output}_rotation_{i}.pt"
                        )
                        if os.path.exists(g_path) and os.path.exists(l_path):
                            file_pairs.append((g_path, l_path))

    print(f"Found {len(file_pairs)} graphs.")
    if args.split_mode == "ic":
        train_pairs, val_pairs, test_pairs = split_by_simulation(file_pairs, 0.8, 0.1, seed=42)
        train_set = LazyGraphDataset(train_pairs)
        val_set = LazyGraphDataset(val_pairs)
        test_set = LazyGraphDataset(test_pairs)
    else:
        full_dataset = LazyGraphDataset(file_pairs)
        train_set, val_set, test_set = random_split(
            full_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    model = GCN().to(device)
    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        print(f"Loading pre-trained weights from '{args.pretrained_weights}'...")
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    pos_weight_tensor = torch.tensor([args.pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(
            f"=== START OF GCN TRAINING (split={args.split_mode}, pretrained={args.pretrained_weights}) ===\n"
        )
        f.write("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 | Val BalAcc\n")

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        total_nodes_train = 0
        tp_tr = fp_tr = tn_tr = fn_tr = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch.num_nodes
            preds = (out > 0.0).float()
            tp_tr += int(((preds == 1) & (batch.y == 1)).sum().item())
            fp_tr += int(((preds == 1) & (batch.y == 0)).sum().item())
            tn_tr += int(((preds == 0) & (batch.y == 0)).sum().item())
            fn_tr += int(((preds == 0) & (batch.y == 1)).sum().item())
            total_nodes_train += batch.num_nodes

        model.eval()
        running_val_loss = 0.0
        total_nodes_val = 0
        tp_val = fp_val = tn_val = fn_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch).view(-1)
                loss = criterion(out, batch.y)
                running_val_loss += loss.item() * batch.num_nodes
                preds = (out > 0.0).float()
                tp_val += int(((preds == 1) & (batch.y == 1)).sum().item())
                fp_val += int(((preds == 1) & (batch.y == 0)).sum().item())
                tn_val += int(((preds == 0) & (batch.y == 0)).sum().item())
                fn_val += int(((preds == 0) & (batch.y == 1)).sum().item())
                total_nodes_val += batch.num_nodes

        avg_train_loss = running_train_loss / max(1, total_nodes_train)
        avg_val_loss = running_val_loss / max(1, total_nodes_val)
        train_acc, _, _, _, _, _ = compute_binary_metrics(tp_tr, fp_tr, tn_tr, fn_tr)
        val_acc, _, _, val_f1, val_bal_acc, _ = compute_binary_metrics(
            tp_val, fp_val, tn_val, fn_val
        )

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_model)

        log_line = (
            f"{epoch+1:03d}   | {avg_train_loss:.4f}     | {avg_val_loss:.4f}   | "
            f"{train_acc:.2f}%    | {val_acc:.2f}%  | {val_f1:.4f} | {val_bal_acc:.2f}%"
        )
        print(log_line)
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=device))
    model.eval()
    running_test_loss = 0.0
    total_nodes_test = 0
    tp_te = fp_te = tn_te = fn_te = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            loss = criterion(out, batch.y)
            running_test_loss += loss.item() * batch.num_nodes
            preds = (out > 0.0).float()
            tp_te += int(((preds == 1) & (batch.y == 1)).sum().item())
            fp_te += int(((preds == 1) & (batch.y == 0)).sum().item())
            tn_te += int(((preds == 0) & (batch.y == 0)).sum().item())
            fn_te += int(((preds == 0) & (batch.y == 1)).sum().item())
            total_nodes_test += batch.num_nodes

    avg_test_loss = running_test_loss / max(1, total_nodes_test)
    test_acc, test_prec, test_rec, test_f1, test_bal_acc, test_mcc = compute_binary_metrics(
        tp_te, fp_te, tn_te, fn_te
    )
    summary_str = (
        f"Final Test Loss: {avg_test_loss:.4f} | Acc: {test_acc:.2f}% | "
        f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} | "
        f"Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | MCC: {test_mcc:.4f}"
    )
    print(summary_str)
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write("-" * 75 + "\n")
        f.write(summary_str + "\n")
        f.write("=== END OF GCN TRAINING ===\n")
