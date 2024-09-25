import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os

from .eval import evaluate

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    CE_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-CE_loss)
    F_loss = alpha * (1-pt)**gamma * CE_loss
    return F_loss.mean()

def save_checkpoint(epoch, model, optimizer, train_loss_history, val_loss_history, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
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
    print(f"Checkpoint loaded from {checkpoint_path}, Epoch: {epoch}")
    return epoch, train_loss_history, val_loss_history

def train(model, train_loader, eval_loader, epochs, lr, device, 
          alpha=0.25, gamma=2, checkpoint_dir="checkpoints", start_epoch=0, train_loss_history=None, val_loss_history=None):
    
    # Initial learning rate with scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    criterion = focal_loss
    
    if train_loss_history is None:
        train_loss_history = []
    if val_loss_history is None:
        val_loss_history = []
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, alpha=alpha, gamma=gamma)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        train_loss_history.append(avg_train_loss)
        
        avg_val_loss = evaluate(model, eval_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}")
        val_loss_history.append(avg_val_loss)
        
        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, train_loss_history, val_loss_history, checkpoint_dir)
        
    return model, train_loss_history, val_loss_history