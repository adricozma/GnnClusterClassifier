import time
import matplotlib.pyplot as plt
import os

log_filename = "training_metrics_GNN_02.log"

while True:
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
        
        if epochs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Subplot 1 : Losses
            ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
            ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
            ax1.set_title('Loss Evolution')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            ax1.legend()
            
            # Subplot 2 : Accuracies
            ax2.plot(epochs, train_accs, 'b-s', label='Train Acc')
            ax2.plot(epochs, val_accs, 'r-s', label='Val Acc')
            ax2.set_title('Accuracy Evolution')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig("live_plots.png")
            plt.close()
            print(f"Graphs refreshed at epoch {epochs[-1]} - Image saved.")
            
    time.sleep(100)