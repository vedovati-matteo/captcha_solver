import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, OneCycleLR
from torchvision import models
from torchvision.transforms import transforms
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(CharClass())
num_epochs = 50

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_layers(model, unfreeze_last_n):
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last n layers
    for param in list(model.parameters())[-unfreeze_last_n:]:
        param.requires_grad = True

def get_model(model_name, num_classes, seed):
    set_seed(seed)
    
    if model_name == 'efficientnet':
        model = models.efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    freeze_layers(model, unfreeze_last_n=5)  # Adjust the number as needed
    return model

# Create the ensemble
models = [
    get_model('efficientnet', num_classes, seed=42),
    get_model('resnet', num_classes, seed=43),
    get_model('densenet', num_classes, seed=44),
    get_model('mobilenet', num_classes, seed=45),
    get_model('inception', num_classes, seed=46)
]

# Define different optimizers, schedulers, and loss functions for each model
optimizers = [
    torch.optim.Adam(models[0].parameters(), lr=1e-4),
    torch.optim.SGD(models[1].parameters(), lr=1e-3, momentum=0.9),
    torch.optim.RMSprop(models[2].parameters(), lr=1e-4),
    torch.optim.AdamW(models[3].parameters(), lr=1e-4),
    torch.optim.Adagrad(models[4].parameters(), lr=1e-3)
]

# Define different schedulers for each model
schedulers = [
    CosineAnnealingLR(optimizers[0], T_max=num_epochs),
    StepLR(optimizers[1], step_size=10, gamma=0.1),
    ExponentialLR(optimizers[2], gamma=0.95),
    OneCycleLR(optimizers[3], max_lr=1e-3, total_steps=num_epochs),
    lambda epoch: 1.0  # Constant learning rate for the last model
]

# Custom Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Define different loss functions for each model
loss_functions = [
    nn.CrossEntropyLoss(),
    FocalLoss(alpha=1, gamma=2),
    nn.CrossEntropyLoss(label_smoothing=0.1),
    lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets) + 1e-5 * sum(p.pow(2.0).sum() for p in models[3].parameters()),
    nn.CrossEntropyLoss(weight=torch.ones(num_classes))  # You can adjust weights if classes are imbalanced
]

# Define different data augmentation techniques for each model
def get_augmentation(augmentation_type):
    if augmentation_type == 'rotate_flip':
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
        ])
    elif augmentation_type == 'color_jitter':
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    elif augmentation_type == 'noise_blur':
        return transforms.Compose([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            lambda x: x + torch.randn_like(x) * 0.1,
        ])
    elif augmentation_type == 'perspective':
        return transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        ])
    elif augmentation_type == 'mixup':
        return transforms.Compose([
            transforms.MixUp(alpha=0.2),
        ])
    else:
        return transforms.Compose([])  # No augmentation

augmentation_techniques = [
    get_augmentation('rotate_flip'),
    get_augmentation('color_jitter'),
    get_augmentation('noise_blur'),
    get_augmentation('perspective'),
    get_augmentation('mixup')
]

datatset_path = '../datasets/dataset_v3'
batch_size = 256

seed = 23
torch.manual_seed(seed)
random.seed(seed)

dataset = CharDataset(datatset_path, transform=transform, max_crop_error=0.1)

total_size = len(dataset)
train_size = int(0.90 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate, prefetch_factor=2)
# Create the DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate, prefetch_factor=2)

# Training loop
for epoch in range(num_epochs):
    for i, model in enumerate(models):
        model.train()
        running_loss = 0.0
        
        # Get a subset of the data for this model
        subset = get_data_subset(train_data, fraction=0.9)  # Implement this function to get a random subset
        
        for inputs, labels in subset:
            # Apply specific data augmentation for this model
            inputs = augmentation_techniques[i](inputs)
            
            # Move to device (GPU if available)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizers[i].zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_functions[i](outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizers[i].step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f'Model {i+1}, Epoch {epoch+1}, Loss: {running_loss / len(subset):.4f}')
        
        # Update learning rate
        if i != 4:  # Skip for the last model with constant learning rate
            schedulers[i].step()
    
    # Validate the ensemble (implement this function)
    validate_ensemble(models, val_loader)

# Save the trained models
for i, model in enumerate(models):
    torch.save(model.state_dict(), f'model_{i+1}.pth')

print("Training complete!")