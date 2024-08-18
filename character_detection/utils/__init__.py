from PIL import Image
from PIL.ImageDraw import Draw
from matplotlib import pyplot as plt
import torch

from .data_utils import ResizeWithPad, RPNDataset, pad_value, transform, transform_noise, collate_fn
from .train_utils import box_iou, compute_loss
from .eval_utils import calculate_ap, calculate_precision_recall_f1


def show_image(image_path, target_bb):
    image = Image.open(image_path).convert("RGB")
    image = ResizeWithPad(224)(image)

    draw = Draw(image)
    for bb in target_bb:
        draw.rectangle(bb, outline=(255, 0, 0), width=2)
    image.show()
    
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

def plot_precision_recall_f1(precision_recall_history, ax=None):
    precisions, recalls, f1s = zip(*precision_recall_history)
    epochs = range(1, len(precisions) + 1)
    
    if ax is None:
        plt.figure(figsize=(10, 5))
        plt.plot(precisions, label="Precision")
        plt.plot(recalls, label="Recall")
        plt.plot(f1s, label="F1-score")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.show()
    else:
        ax.clear()
        ax.plot(epochs, precisions, label='Precision')
        ax.plot(epochs, recalls, label='Recall')
        ax.plot(epochs, f1s, label='F1-Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.legend()
        ax.set_title('Precision, Recall, and F1-Score Over Epochs')
        ax.figure.canvas.draw()  # Update the plot

def seve_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model