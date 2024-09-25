from matplotlib import pyplot as plt

from .data_utils import CharClass, CharDataset, transform, transform_noise, custom_collate, pad_to_square

def plot_loss(train_loss_history, val_loss_history, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        ax.clear()
        ax.plot(train_loss_history, label='Train Loss')
        ax.plot(val_loss_history, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Train and Validation Loss Over Epochs')
        ax.figure.canvas.draw()  # Update the plot