"""Training and curriculum fine-tuning script for the 3-layer Residual GCN classifier.

Supports:
- Simulation-level (`--split_mode ic`) train/val/test partitioning (zero data leakage).
- Local k-NN phase-space relational feature encoding (`src/gnn_models.py`).
- Two-stage Curriculum Learning (`--pretrained_weights`).
- Arbitrary observable subsets (`--feature_indices`, e.g. 5D or 4D ablation studies).
- Real-time epoch-by-epoch live training curve generation (`scripts/live_plot_utils.py`).
"""

import argparse
import os
import subprocess
import sys
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_models import GCN
from live_plot_utils import update_live_plots_for_log

parser = argparse.ArgumentParser(description="Train or fine-tune the 3-layer Residual GCN model.")
parser.add_argument("--data_folder", "--data_dir", dest="data_folder", type=str, required=True, help="Path to graph data folder.")
parser.add_argument("--n", type=int, default=5, help="Number of 3D rotations per snapshot.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=256, help="Mini-batch size.")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--log_file", type=str, default="training_metrics_gcn.log", help="Log file path.")
parser.add_argument(
    "--split_mode",
    type=str,
    default="ic",
    choices=["ic", "file", "random"],
    help="Split dataset by stream ID ('ic', no data leakage) or by individual file ('file', legacy).",
)
parser.add_argument(
    "--pos_weight",
    type=float,
    default=1.0,
    help="Positive class weight for BCEWithLogitsLoss.",
)
parser.add_argument(
    "--pretrained_weights",
    type=str,
    default="",
    help="Optional path to pre-trained model weights (.pth) for Stage 2 curriculum fine-tuning.",
)
parser.add_argument(
    "--save_model",
    type=str,
    default="best_gcn_model.pth",
    help="Output path to save the best model checkpoint (.pth).",
)
parser.add_argument(
    "--feature_indices",
    type=str,
    default="",
    help="Optional comma-separated feature column indices to select on the fly (e.g. '0,1,3,4' for 4-observable ablation).",
)

args = parser.parse_args()

data_folder = args.data_folder
n = args.n
Lr = args.lr
bs = args.bs
epochs = args.epochs
log_filename = args.log_file
split_mode = args.split_mode
pos_weight_val = args.pos_weight
pretrained_weights = args.pretrained_weights
best_model_path = args.save_model
feature_indices = (
    [int(x.strip()) for x in args.feature_indices.split(",") if x.strip() != ""]
    if args.feature_indices
    else None
)

os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)

# ---------- CPU / DEVICE MANAGEMENT ----------
num_cpus = int(os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_CPUS_PER_TASK", 72)))
math_threads = min(max(1, num_cpus), 72)
torch.set_num_threads(math_threads)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | PyTorch configured to use {torch.get_num_threads()} CPU cores")


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


outputlist = list(range(0, 101))
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

    print(f"Stream-level split: {len(train_ics)} Train ICs, {len(val_ics)} Val ICs, {len(test_ics)} Test ICs")
    print(f"  Val ICs : {val_ics}")
    print(f"  Test ICs: {test_ics}")

    train_set = AstroGraphDataset(collect_files_for_ics(train_ics), feat_idx=feature_indices)
    val_set = AstroGraphDataset(collect_files_for_ics(val_ics), feat_idx=feature_indices)
    test_set = AstroGraphDataset(collect_files_for_ics(test_ics), feat_idx=feature_indices)
else:
    file_paths = collect_files_for_ics(ic_dirs)
    print(f"End of indexing! {len(file_paths)} files found.")
    dataset = AstroGraphDataset(file_paths, feat_idx=feature_indices)
    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

sample_graph = train_set[0]
in_channels = int(sample_graph.x.shape[1])
print(f"Detected input node feature dimension: in_channels = {in_channels}")

nb_workers = min(12, max(2, math_threads // 6))
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nb_workers)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nb_workers)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nb_workers)

with open(log_filename, "w", encoding="utf-8") as f:
    f.write(
        f"=== START of training (split_mode={split_mode}, in_channels={in_channels}, pos_weight={pos_weight_val}) ===\n"
    )
    if split_mode == "ic":
        f.write(f"Val ICs : {val_ics}\n")
        f.write(f"Test ICs: {test_ics}\n")
    f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR\n")

model = GCN(in_channels=in_channels).to(device)
if pretrained_weights:
    if not os.path.exists(pretrained_weights):
        raise FileNotFoundError(f"Pre-trained weights file not found: {pretrained_weights}")
    print(f"Loading pre-trained weights from '{pretrained_weights}' for fine-tuning...")
    model.load_state_dict(torch.load(pretrained_weights, map_location=device, weights_only=True))

optimizer = torch.optim.AdamW(model.parameters(), lr=Lr, weight_decay=1e-4)
if pos_weight_val != 1.0:
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
else:
    criterion = torch.nn.BCEWithLogitsLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_nodes_train = 0
    running_train_loss = 0.0
    running_train_correct = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).view(-1)

        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_train_loss += loss.item() * batch.num_nodes
        total_nodes_train += batch.num_nodes

        preds = (torch.sigmoid(out) > 0.5).float()
        running_train_correct += (preds == batch.y).sum().item()

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch {epoch+1:03d} | Batch {batch_idx+1:04d}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

    model.eval()
    total_nodes_val = 0
    running_val_loss = 0.0
    running_val_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            loss = criterion(out, batch.y)

            running_val_loss += loss.item() * batch.num_nodes
            total_nodes_val += batch.num_nodes

            preds = (torch.sigmoid(out) > 0.5).float()
            running_val_correct += (preds == batch.y).sum().item()

    avg_train_loss = running_train_loss / max(1, total_nodes_train)
    avg_val_loss = running_val_loss / max(1, total_nodes_val)
    train_acc = (running_train_correct / max(1, total_nodes_train)) * 100
    val_acc = (running_val_correct / max(1, total_nodes_val)) * 100
    current_lr = optimizer.param_groups[0]["lr"]

    log_line = (
        f"{epoch+1:03d}    | {avg_train_loss:.4f}     | {train_acc:.2f}%   | {avg_val_loss:.4f}   | {val_acc:.2f}%  | {current_lr:.1e}"
    )
    print("-" * 80)
    print(f"FIN EPOCH {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%) | LR: {current_lr:.1e}")
    print("-" * 80)

    scheduler.step(avg_val_loss)

    is_best = False
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print("      --> Best model saved!")
        log_line += " (*)"
        is_best = True

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

print("\nFinal test evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
model.eval()
total_nodes_test = 0
running_test_loss = 0.0
running_test_correct = 0
tp_te = fp_te = tn_te = fn_te = 0

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        loss = criterion(out, batch.y)

        running_test_loss += loss.item() * batch.num_nodes
        total_nodes_test += batch.num_nodes

        preds = (torch.sigmoid(out) > 0.5).float()
        running_test_correct += (preds == batch.y).sum().item()
        tp_te += int(((preds == 1) & (batch.y == 1)).sum().item())
        fp_te += int(((preds == 1) & (batch.y == 0)).sum().item())
        tn_te += int(((preds == 0) & (batch.y == 0)).sum().item())
        fn_te += int(((preds == 0) & (batch.y == 1)).sum().item())

avg_test_loss = running_test_loss / max(1, total_nodes_test)
test_acc = (running_test_correct / max(1, total_nodes_test)) * 100
test_prec = tp_te / max(1, tp_te + fp_te)
test_rec = tp_te / max(1, tp_te + fn_te)
test_spec = tn_te / max(1, tn_te + fp_te)
test_f1 = 2.0 * test_prec * test_rec / max(1e-12, test_prec + test_rec)
test_bal_acc = 50.0 * (test_rec + test_spec)

print(
    f"--> Final Test Loss: {avg_test_loss:.4f} | Final Test Acc: {test_acc:.2f}% | "
    f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} (Prec: {test_prec:.4f}, Rec: {test_rec:.4f})"
)

with open(log_filename, "a", encoding="utf-8") as f:
    f.write("-" * 50 + "\n")
    f.write(
        f"Final Test Loss : {avg_test_loss:.4f} | Acc: {test_acc:.2f}% | "
        f"BalAcc: {test_bal_acc:.2f}% | F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f}\n"
    )
    f.write("=== END OF TRAINING ===\n")

try:
    update_live_plots_for_log(log_filename)
    subprocess.run(
        [sys.executable, os.path.join(ROOT_DIR, "scripts", "live_monitor.py"), "--once"],
        check=False,
        timeout=30,
    )
except Exception:
    pass

print(f"Training finished. Logs written to '{log_filename}'. Best model is '{best_model_path}'.")
