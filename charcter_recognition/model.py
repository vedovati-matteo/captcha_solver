
import torch
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torch.nn as nn

from utils import CharClass

class CaptchaCharClassifier(nn.Module):
    def __init__(self, dropout_rate, device):
        super(CaptchaCharClassifier, self).__init__()
        self.efficientnet = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        num_classes = len(CharClass())
        
        # Remove the original classifier
        self.features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Freeze the first half of the layers
        num_layers = len(list(self.features.parameters()))
        for param in list(self.features.parameters())[:4 * num_layers // 5]: # 80 % of the layers
            param.requires_grad = False
        
        # Add dropout and new classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

