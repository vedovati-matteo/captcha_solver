
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

from .utils import CharClass

class CharRecognizer(nn.Module):
    def __init__(self, dropout_rate, device, temperature=1.0):
        super(CharRecognizer, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_classes = len(CharClass())
        
        # Remove the original classifier
        self.features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Freeze the first half of the layers
        num_layers = len(list(self.features.parameters()))
        for param in list(self.features.parameters())[:1 * num_layers // 4]: # 50 % of the layers
            param.requires_grad = False
        
        # Add dropout and new classifier with an additional hidden layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x / self.temperature

