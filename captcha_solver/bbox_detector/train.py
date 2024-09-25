import torch
import os
from tqdm import tqdm

from .eval import evaluate
from .utils import compute_loss

def save_checkpoint(epoch, model, optimizer, train_loss_history, val_loss_history, precision_recall_history, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'precision_recall_history': precision_recall_history
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, weights_only=True, optimizer=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    if weights_only:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Weights loaded from {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']
    precision_recall_history = checkpoint['precision_recall_history']

    print(f"Checkpoint loaded from {checkpoint_path}, Epoch: {epoch}")

    return epoch, train_loss_history, val_loss_history, precision_recall_history

def train(model, train_loader, eval_loader, epochs, lr, conf_threshold, nms_threshold, device, checkpoint_dir="checkpoints", start_epoch=0, train_loss_history=None, val_loss_history=None, precision_recall_history=None):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if train_loss_history is None:
        train_loss_history = []
    if val_loss_history is None:
        val_loss_history = []
    if precision_recall_history is None:
        precision_recall_history = []
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            proposals, objectness_logits = model(images)
            
            # Compute loss here
            loss, classification_loss, bbox_regression_loss = compute_loss(proposals, targets, objectness_logits)
            
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        train_loss_history.append(avg_train_loss)        
        
        avg_val_loss, mAP, avg_precision, avg_recall, avg_f1, = evaluate(model, eval_loader, conf_threshold, nms_threshold, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}, mAP: {mAP:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-score: {avg_f1:.4f}")
        val_loss_history.append(avg_val_loss)
        precision_recall_history.append((avg_precision, avg_recall, avg_f1))

        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, train_loss_history, val_loss_history, precision_recall_history, checkpoint_dir)
        
    return model, train_loss_history, val_loss_history, precision_recall_history
        
        