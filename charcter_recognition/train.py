import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from eval import evaluate
from utils import CharClass

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.1, num_classes=10):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate cross entropy with label smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(true_dist * log_probs).sum(dim=1)

        # Calculate focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        return focal_loss.mean()

def save_checkpoint(epoch, model, optimizer,  scheduler, train_loss_history, val_loss_history, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: Scheduler state not found in checkpoint. Initializing a new scheduler.")
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    epoch = checkpoint['epoch']
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']

    print(f"Checkpoint loaded from {checkpoint_path}, Epoch: {epoch}")

    return epoch, train_loss_history, val_loss_history, scheduler

def mixup_data(x, y, alpha=0.2):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, train_loader, eval_loader, epochs, lr, device, 
          checkpoint_dir="checkpoints", start_epoch=0, train_loss_history=None, val_loss_history=None, 
          mixup_alpha=0.4, mixup_prob=0.5, T_0=10, T_mult=2, scheduler=None):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLossWithLabelSmoothing(num_classes=len(CharClass()))
    
    if scheduler is None:
        # Initialize the learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    
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
            
            # Decide whether to apply mixup for this batch
            use_mixup = np.random.random() < mixup_prob
            
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
                inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Step the scheduler
        scheduler.step()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        train_loss_history.append(avg_train_loss)
        
        avg_val_loss = evaluate(model, eval_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}")
        val_loss_history.append(avg_val_loss)
        
        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, scheduler, train_loss_history, val_loss_history, checkpoint_dir)
        
    return model, train_loss_history, val_loss_history