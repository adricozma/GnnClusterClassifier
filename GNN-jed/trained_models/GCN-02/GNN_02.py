import argparse  
import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# ---------- PARSER ----------
parser = argparse.ArgumentParser(
    description="Train GAT model with custom parameters"
)
parser.add_argument(
    "--data_folder", type=str, required=True, help="Path to data folder"
)
parser.add_argument(
    "--n", type=int, required=True, help="Number of rotations"
)
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--bs", type=int, required=True, help="Batch size")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
parser.add_argument("--log_file", type=str, default="training_metrics.log", help="Log file name")
parser.add_argument(
    "--split_mode",
    type=str,
    default="ic",
    choices=["ic", "file"],
    help="Split dataset by stream ID ('ic', no data leakage) or by individual file ('file', legacy)",
)
parser.add_argument(
    "--pos_weight",
    type=float,
    default=1.0,
    help="Positive class weight for BCEWithLogitsLoss to handle class imbalance (e.g. ~2.9)",
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
# -------------------------------------------------------

# ---------- INTERN: CPU MANAGEMENT ----------
num_cpus = int(os.environ.get("SLURM_NTASKS", 1))
math_threads = min(num_cpus, 72)
torch.set_num_threads(math_threads)
print(f" PyTorch configured to use {torch.get_num_threads()} CPU cores")
# -----------------------------------------------------------------


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GATConv(5, 128)
        self.conv2 = GATConv(128, 256)
        self.conv3 = GATConv(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x
    

class AstroGraphDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        data = torch.load(filepath, weights_only=False)
        
        label_path = filepath.replace("graph_data_output", "label_output")
        target = torch.load(label_path, weights_only=False)
        
        data.y = target.float()
        return data
    
outputlist = list(range(0, 101))

print(f"Indexing the files in the data folder (split_mode='{split_mode}')...")

ic_dirs = sorted([d for d in os.listdir(data_folder) if d.startswith("IC_")])

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
    # Split at the stream (IC_ID) level so all snapshots and rotations of a given stream
    # stay strictly in Train, Val, or Test (zero spatio-temporal or rotational data leakage)
    train_ics_subset, val_ics_subset, test_ics_subset = random_split(
        ic_dirs, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )
    train_ics = [ic_dirs[i] for i in train_ics_subset.indices]
    val_ics   = [ic_dirs[i] for i in val_ics_subset.indices]
    test_ics  = [ic_dirs[i] for i in test_ics_subset.indices]

    print(f"Stream-level split: {len(train_ics)} Train ICs, {len(val_ics)} Val ICs, {len(test_ics)} Test ICs")
    print(f"  Val ICs : {val_ics}")
    print(f"  Test ICs: {test_ics}")

    train_set = AstroGraphDataset(collect_files_for_ics(train_ics))
    val_set   = AstroGraphDataset(collect_files_for_ics(val_ics))
    test_set  = AstroGraphDataset(collect_files_for_ics(test_ics))
else:
    file_paths = collect_files_for_ics(ic_dirs)
    print(f"End of indexing ! {len(file_paths)} files found.")
    dataset = AstroGraphDataset(file_paths)
    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

total_nodes_train = sum(d.num_nodes for d in train_set)
total_nodes_val = sum(d.num_nodes for d in val_set)
total_nodes_test = sum(d.num_nodes for d in test_set)

# they load the data in // to improve training speed 
nb_workers = 12

train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nb_workers)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nb_workers)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nb_workers)

with open(log_filename, "w", encoding="utf-8") as f:
    f.write(f"=== START of training (split_mode={split_mode}, pos_weight={pos_weight_val}) ===\n")
    if split_mode == "ic":
        f.write(f"Val ICs : {val_ics}\n")
        f.write(f"Test ICs: {test_ics}\n")
    f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR\n")

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
if pos_weight_val != 1.0:
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
else:
    criterion = torch.nn.BCEWithLogitsLoss()

# Learning rate scheduler (reduces LR by factor of 10 if val_loss plateaus for 3 epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Variables to track the best model for checkpointing
best_val_loss = float('inf')
best_model_path = "best_gnn_model_02.pth"

train_loss_history = []
val_loss_history = []

for epoch in range(epochs):

    model.train()
    total_train_loss = 0
    running_train_loss = 0.0
    # Variable to track correct predictions for training accuracy
    running_train_correct = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch)
        out = out.view(-1)

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch.num_nodes
        
        # Calculate correct predictions (binary classification threshold = 0.5)
        preds = (torch.sigmoid(out) > 0.5).float()
        running_train_correct += (preds == batch.y).sum().item()

        print(f"Epoch {epoch+1:03d} | Batch {batch_idx+1:04d}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

    model.eval()
    total_val_loss = 0
    running_val_loss = 0.0
    # Variable to track correct predictions for validation accuracy
    running_val_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            out = out.view(-1)
            loss = criterion(out, batch.y)

            running_val_loss += loss.item() * batch.num_nodes
            
            # Calculate correct predictions for validation
            preds = (torch.sigmoid(out) > 0.5).float()
            running_val_correct += (preds == batch.y).sum().item()

    avg_train_loss = running_train_loss / total_nodes_train
    avg_val_loss = running_val_loss / total_nodes_val
    
    # Calculate final accuracy percentages
    train_acc = (running_train_correct / total_nodes_train) * 100
    val_acc = (running_val_correct / total_nodes_val) * 100

    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)

    # Extract current learning rate for logging
    current_lr = optimizer.param_groups[0]['lr']

    # Updated log string with new metrics
    log_line = (
            f"{epoch+1:03d}    | {avg_train_loss:.4f}     | {train_acc:.2f}%   | {avg_val_loss:.4f}   | {val_acc:.2f}%  | {current_lr:.1e}"
    )
    print("-" * 80)
    print(f"FIN EPOCH {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%) | LR: {current_lr:.1e}")
    print("-" * 80)

    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)

    #Save the model if validation loss improves (Model Checkpoint)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print("      --> Best model saved!")
        log_line += " (*)"

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    


print("\nfinal evaluation...")

# Load the best model weights before final testing
print("Loading best model weights for test evaluation...")
model.load_state_dict(torch.load(best_model_path))

model.eval()
running_test_loss = 0.0
running_test_correct = 0

with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        out = out.view(-1)
        loss = criterion(out, batch.y)

        running_test_loss += loss.item() * batch.num_nodes
        
        preds = (torch.sigmoid(out) > 0.5).float()
        running_test_correct += (preds == batch.y).sum().item()

avg_test_loss = running_test_loss / total_nodes_test

# Calculate test accuracy
test_acc = (running_test_correct / total_nodes_test) * 100

print(f"--> Final Test Loss: {avg_test_loss:.4f} | Final Test Acc: {test_acc:.2f}%")

with open(log_filename, "a", encoding="utf-8") as f:
    f.write("-" * 50 + "\n")
    f.write(f"Final Test Loss : {avg_test_loss:.4f} | Acc: {test_acc:.2f}%\n")
    f.write("=== END OF TRAINING ===\n")

# best model is already saved during the loop.
print(f"Training finished. Logs written to '{log_filename}'. Best model is '{best_model_path}'.")