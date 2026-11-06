import time
import matplotlib.pyplot as plt
import os
from adri_plotstyle import init_plot_style
init_plot_style(14)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODELS_DIR = os.path.join(ROOT_DIR, "GNN-jed", "trained_models")

log_filenames = [
    os.path.join(TRAINED_MODELS_DIR, "GCN-01", "training_metrics_GNN_01.log"),
    os.path.join(TRAINED_MODELS_DIR, "GCN-02", "training_metrics_GNN_02.log"),
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
i = 0
for log_filename in log_filenames:
    i += 1
    print(log_filename)
    if os.path.exists(log_filename):
        epochs, train_losses, train_accs, val_losses, val_accs = [], [], [], [], []
        
        with open(log_filename, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            if "|" in line and "Train Loss" not in line and "START" not in line:
                parts = line.split("|")
                try:
                    epochs.append(int(parts[0].strip()))
                    train_losses.append(float(parts[1].strip()))
                    train_accs.append(float(parts[2].replace("%","").strip()))
                    val_losses.append(float(parts[3].strip()))
                    val_accs.append(float(parts[4].replace("%","").strip()))
                except ValueError:
                    continue 
        line_width = 1.5
        marker = 'o' if i == 1 else 's'
        marker_size = 4
        if epochs:
            color_val = 'r' if i == 1 else 'g'
            color_train = 'b' if i == 1 else 'k'
            
            # Subplot 1 : Losses
            los_train_label = r'$\mathrm{Train\ Loss\ GCN}$' if i == 1 else r'$\mathrm{Train\ Loss\ GAT)}$'
            los_val_label = r'$\mathrm{Val\ Loss\ GCN}$' if i == 1 else r'$\mathrm{Val\ Loss\ GAT}$'

            ax1.plot(epochs, train_losses, color_train + '-'+marker, label=los_train_label, markersize=marker_size, linewidth=line_width)
            ax1.plot(epochs, val_losses, color_val + '-'+marker, label=los_val_label, markersize=marker_size, linewidth=line_width)
            #ax1.set_title(r'$\mathrm{Loss\ Evolution}$')
            ax1.set_xlabel(r'$\mathrm{Epoch}$')
            ax1.set_ylabel(r'$\mathrm{Loss}$')
            ax1.grid(True)
            ax1.legend()
            
            # Subplot 2 : Accuracies
            acc_train_label = r'$\mathrm{Train\ Acc\ GCN}$' if i == 1 else r'$\mathrm{Train\ Acc\ GAT}$'
            acc_val_label = r'$\mathrm{Val\ Acc\ GCN}$' if i == 1 else r'$\mathrm{Val\ Acc\ GAT}$'

            ax2.plot(epochs, train_accs, color_train + '-'+marker, label=acc_train_label, markersize=marker_size, linewidth=line_width)
            ax2.plot(epochs, val_accs, color_val + '-'+marker, label=acc_val_label, markersize=marker_size, linewidth=line_width)
            #ax2.set_title(r'$\mathrm{Accuracy\ Evolution}$')    
            ax2.set_xlabel(r'$\mathrm{Epoch}$')
            ax2.set_ylabel(r'$\mathrm{Accuracy\ (\%)}$')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
plt.savefig("live_plots_GNN.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Graphs refreshed at epoch {epochs[-1]} - Image saved.")


# log_filename = "GCN-02/job_GNN_02_54597602.out"

# if os.path.exists(log_filename):
#     batch_epochs, batch_losses = [], []
#     epoch_x, val_losses, train_accs, val_accs = [], [], [], []
    
#     with open(log_filename, "r") as f:
#         lines = f.readlines()
        
#     for line in lines:
#         if line.startswith("Epoch ") and "Batch Loss" in line:
#             try:
#                 parts = line.split("|")
#                 ep = int(parts[0].replace("Epoch", "").strip())
#                 b_curr, b_tot = map(int, parts[1].replace("Batch", "").strip().split('/'))
#                 loss = float(parts[2].replace("Batch Loss:", "").strip())
                
#                 frac_ep = ep - 1 + (b_curr / b_tot)
#                 batch_epochs.append(frac_ep)
#                 batch_losses.append(loss)
#             except (ValueError, IndexError):
#                 continue
                
#         elif line.startswith("FIN EPOCH"):
#             try:
#                 parts = line.split("|")
#                 ep = int(parts[0].replace("FIN EPOCH", "").strip())
                
#                 t_loss_acc = parts[1].split("(")
#                 t_acc = float(t_loss_acc[1].replace("Acc:", "").replace("%)", "").strip())
                
#                 v_loss_acc = parts[2].split("(")
#                 v_loss = float(v_loss_acc[0].replace("Val Loss:", "").strip())
#                 v_acc = float(v_loss_acc[1].replace("Acc:", "").replace("%)", "").strip())
                
#                 epoch_x.append(ep)
#                 val_losses.append(v_loss)
#                 train_accs.append(t_acc)
#                 val_accs.append(v_acc)
#             except (ValueError, IndexError):
#                 continue 

#     if batch_epochs:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
#         ax1.plot(batch_epochs, batch_losses, color='blue', alpha=0.4, label='Train Loss (Batch)')
#         if epoch_x:
#             ax1.plot(epoch_x, val_losses, 'r-o', linewidth=2, label='Val Loss (Epoch)')
#         ax1.set_title('Loss Evolution')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Loss')
#         ax1.grid(True)
#         ax1.legend()
        
#         if epoch_x:
#             ax2.plot(epoch_x, train_accs, 'b-s', linewidth=2, label='Train Acc (Epoch)')
#             ax2.plot(epoch_x, val_accs, 'r-s', linewidth=2, label='Val Acc (Epoch)')
#         ax2.set_title('Accuracy Evolution')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Accuracy (%)')
#         ax2.grid(True)
#         ax2.legend()
        
#         plt.tight_layout()
#         plt.savefig("live_plots_detailed_GCN-02.png")
#         plt.close()
        
#         last_ep = epoch_x[-1] if epoch_x else int(batch_epochs[-1])
#         print(f"Detailed graphs refreshed at epoch {last_ep} - Image saved.")