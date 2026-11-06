#%%  ---------- IMPORTS ----------
import os
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
#%%  ---------- HYPERPARAMETERS ----------
epochs = 2
Lr = 0.1
bs = 20
#%% ---------- DEFINE THE GCN MODEL  ----------
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, 16)  # 5 features 
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 1) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  
        
        return x
#%% ---------- PREPARE THE DATASET ----------

dataset = []
outputlist = list(range(0,101))
n = 5
data_folder = '../../../../../../../../Volumes/Adrian 1 SS/MA2/graph_data/'

for dir in os.listdir(data_folder):
    if dir.startswith('IC_'):
        for output in outputlist:
            data_folder_path = data_folder + dir + '/'
            for i in range(n):
                data = torch.load(f'{data_folder_path}/graph_data_output_{output}_rotation_{i}.pt')
                target = torch.load(f'{data_folder_path}/label_output_{output}_rotation_{i}.pt')
                data.y = target.float()
                dataset.append(data)

# auto separation (80% Train, 10% Val, 10% Test)
train_set, val_set, test_set = random_split(
    dataset, 
    [0.8, 0.1, 0.1],
    generator=torch.Generator().manual_seed(42) # for reproducibility
)

total_nodes_train = sum(d.num_nodes for d in train_set)
total_nodes_val = sum(d.num_nodes for d in val_set)
total_nodes_test = sum(d.num_nodes for d in test_set)

# random loaders 
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)


# logging
log_filename = "training_metrics.log"

with open(log_filename, "w", encoding="utf-8") as f:
    f.write("=== START of training ===\n")
    f.write("Epoch | Train Loss | Val Loss\n")

# instance of the model, optimizer and loss function
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=Lr) # we can experiment with different lrates (hyperparameters)
criterion = torch.nn.BCEWithLogitsLoss()                 # Binary Cross-Entropy with Logits Loss for binary classification

train_loss_history = []
val_loss_history = []

#%%  ---------- RUN THE TRAINING LOOP ----------
for epoch in range(epochs):

    # TRAINING PHASE
    model.train()
    total_train_loss = 0
    running_train_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        out = out.view(-1)

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch.num_nodes

    # VALIDATION PHASE 
    model.eval()
    total_val_loss = 0
    running_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            out = out.view(-1)
            loss = criterion(out, batch.y)

            running_val_loss += loss.item() * batch.num_nodes
    
    avg_train_loss = running_train_loss / total_nodes_train
    avg_val_loss = running_val_loss / total_nodes_val

    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)

    # print in the console and in the logfile 
    log_line = f"{epoch+1:03d}    | {avg_train_loss:.4f}     | {avg_val_loss:.4f}"
    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

# --- Testing Phase ---
print("\nfinal evaluation...")
model.eval()
running_test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        out = out.view(-1)
        loss = criterion(out, batch.y)
        
        running_test_loss += loss.item() * batch.num_graphs

avg_test_loss = running_test_loss / len(test_loader.dataset)

# final test loss output
print(f"--> Final Test Loss: {avg_test_loss:.4f}")

with open(log_filename, "a", encoding="utf-8") as f:
    f.write("-" * 35 + "\n")
    f.write(f"Final Test Loss : {avg_test_loss:.4f}\n")
    f.write("=== END OF TRAINING ===\n")

# Saving the model
torch.save(model.state_dict(), 'gcn_model.pth')
print(f"Model saved and logs written to '{log_filename}'")
# %%


# %% Final evaluation and visualisation of the results for a specific output and rotation (example: output 99, rotation 0)
import matplotlib.pyplot as plt
import torch

model = GCN()
model.load_state_dict(torch.load('gcn_model.pth')) 
model.eval() 

output_target = 90
output_rotation = 0

data = torch.load(f'graph_data/IC_3/graph_data_output_{output_target}_rotation_{output_rotation}.pt')
target_tensor = torch.load(f'graph_data/IC_3/label_output_{output_target}_rotation_{output_rotation}.pt')
true_labels = target_tensor.numpy().astype(int)

with torch.no_grad():
    out = model(data)
    out = out.view(-1)
    probs = torch.sigmoid(out)
    preds = (probs > 0.5).int().numpy()

true_positives = (true_labels == 1) & (preds == 1)
missed_bound = (true_labels == 1) & (preds == 0)
true_negatives = (true_labels == 0) & (preds == 0)
false_positives = (true_labels == 0) & (preds == 1)

ra = data.x[:, 0].numpy()
dec = data.x[:, 1].numpy()

plt.figure(figsize=(8, 6))
plt.scatter(ra[true_negatives], dec[true_negatives], c='lightgray', s=2, alpha=0.3, label='Stream (OK)')
plt.scatter(ra[true_positives], dec[true_positives], c='blue', s=10, alpha=0.6, label='Bound (Correct)', zorder=5)
plt.scatter(ra[missed_bound], dec[missed_bound], c='red', s=20, edgecolors='black', label='MISSING (Bound non detected)')
plt.scatter(ra[false_positives], dec[false_positives], c='orange', s=10, marker='x', label='False Positive', zorder=2)

plt.gca().invert_xaxis() 
plt.xlabel('RA (standardised)')
plt.ylabel('Dec (standardised)')
plt.legend()
plt.show()
# %%
