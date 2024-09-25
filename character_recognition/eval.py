import torch
from tqdm import tqdm

def evaluate(model, val_loader, criterion, device):
    # Evaluation loop
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
    return avg_val_loss