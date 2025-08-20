import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self):
        self.epochs = []
        self.val_r2 = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
    
    def update(self, epoch, train_metrics, val_metrics):
        """Update metrics after each epoch"""
        self.epochs.append(epoch)
        self.val_r2.append(val_metrics['r2'])  # Only validation R²
        self.val_acc.append(val_metrics['accuracy'])
        self.train_loss.append(train_metrics['total_loss'])
        self.val_loss.append(val_metrics['total_loss'])
    
    def plot_metrics(self, save_path='training_metrics.png'):
        """Create comprehensive training visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: R² over time (validation only)
        ax1.plot(self.epochs, self.val_r2, 'r-', label='Validation R²', linewidth=2, marker='o')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Validation R² Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy over time
        ax2.plot(self.epochs, self.val_acc, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Classification Accuracy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Loss over time
        ax3.plot(self.epochs, self.train_loss, 'b-', label='Train Loss', linewidth=2)
        ax3.plot(self.epochs, self.val_loss, 'r-', label='Val Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Total Loss')
        ax3.set_title('Loss Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Log scale for loss
        
        # Plot 4: R² vs Accuracy correlation
        ax4.scatter(self.val_r2, self.val_acc, c=self.epochs, cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Validation R²')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('R² vs Accuracy Correlation')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for epochs
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Epoch')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()