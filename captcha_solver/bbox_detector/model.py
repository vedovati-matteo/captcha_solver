import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class BBoxDetector(nn.Module):
    def __init__(self, dropout_rate, device):
        super(BBoxDetector, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the final layers
        
        # Freeze the ResNet50 layers
        # Convert the parameters generator to a list and freeze all layers except the last two
        backbone_params = list(self.backbone.parameters())
        num_layers = len(list(self.backbone.parameters()))
        for param in backbone_params[:1 * num_layers // 2]: # Adjust to keep the last layer unfrozen
            param.requires_grad = False
        
        self.conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1)  # Adjust input channels to 2048
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified dropout rate
        self.cls_logits = nn.Conv2d(512, 20 * 2, kernel_size=1, stride=1)  # 15 anchors, 2 classes (object/background)
        self.bbox_pred = nn.Conv2d(512, 20 * 4, kernel_size=1, stride=1)  # 15 anchors, 4 bbox coordinates
        self.device = device
    
    def forward(self, x): # [batch_size, 3, 224, 224]
        batch_size, _, _, _ = x.shape
        
        # Pass through the backbone
        features = self.backbone(x)
        conv_features = F.relu(self.conv(features))
        conv_features = self.dropout(conv_features)
        objectness_logits = self.cls_logits(conv_features)  # Predict the probability of an object being present in an anchor box (Shape: [batch_size, 24, H, W])
        # Two classes: object (1) and background (0)
        bbox_deltas = self.bbox_pred(conv_features)         # Refine the anchor boxes to better match the ground truth bounding boxes (Shape: [batch_size, 48, H, W])
        # Four values: (dx, dy, dw, dh)
        
        # Generate anchor boxes
        feature_map_size = conv_features.shape[-2:]  # (H, W) => 7, 7
        stride = x.shape[-2] // feature_map_size[0]  # Assuming square input and feature map (doing it only in one axis) => 224 / 7 = 32
        
        anchors = generate_anchors(feature_map_size, stride, self.device) # [num_anchors (735), 4]
        
        # Reshape objectness_logits [batch_size, num_anchors, 2]
        objectness_logits = objectness_logits.permute(0, 2, 3, 1).contiguous()
        objectness_logits = objectness_logits.view(batch_size, -1, 2)
        
        # Proposals
        proposals = apply_deltas_to_anchors(anchors, bbox_deltas) # [batch_size, num_anchors, 4]
        
        # Return the necessary outputs for loss computation
        # proposals: [batch_size, num_anchors, 4], objectness_logits: [batch_size, num_anchors, 2]
        return proposals, objectness_logits

def generate_anchors(feature_map_size, stride, device):
    # Aspect ratios and scales for the anchor boxes
    # 0.3: 75, 90, 110
    # 0.4: 65, 80, 95
    # 0.5: 55, 70, 85
    # 0.6: 50, 65, 80
    # 0.7: 45, 60, 70
    
    ratios_scale = [[0.3, 70], [0.3, 85], [0.3, 105], [0.3, 120], 
                   [0.4, 60], [0.4, 75], [0.4, 90], [0.4, 105], 
                   [0.5, 50], [0.5, 65], [0.5, 80], [0.5, 95],
                   [0.6, 45], [0.6, 60], [0.6, 75], [0.6, 85],
                   [0.7, 40], [0.7, 55], [0.7, 65], [0.7, 75]]
    
    # Generate the center points of the anchors on the feature map grid
    feature_height, feature_width = feature_map_size
    shifts_x = torch.arange(0, feature_width * stride, stride)
    shifts_y = torch.arange(0, feature_height * stride, stride)
    shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.contiguous().view(-1)
    shift_y = shift_y.contiguous().view(-1)

    # Create a grid of anchor centers
    anchor_centers = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

    # Calculate the anchor sizes and aspect ratios
    anchors = []
    for aspect_ratio, scale in ratios_scale:
        # Compute width and height of the anchor
        anchor_width = scale * torch.sqrt(torch.tensor(aspect_ratio))
        anchor_height = scale / torch.sqrt(torch.tensor(aspect_ratio))

        # Create anchor boxes by adjusting the center points
        anchor_boxes = torch.stack([
            anchor_centers[:, 0] - 0.5 * anchor_width,
            anchor_centers[:, 1] - 0.5 * anchor_height,
            anchor_centers[:, 2] + 0.5 * anchor_width,
            anchor_centers[:, 3] + 0.5 * anchor_height
        ], dim=1)

        anchors.append(anchor_boxes)

    # Concatenate all anchors
    anchors = torch.cat(anchors, dim=0)
    anchors = anchors.to(device)
    
    return anchors

def apply_deltas_to_anchors(anchors, bbox_deltas):
    batch_size = bbox_deltas.shape[0]
    
    # Reshape bbox_deltas to [batch_size, num_anchors, 4]
    bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
    bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

    # Expand anchors to match batch size
    anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract anchor coordinates
    widths = anchors[:, :, 2] - anchors[:, :, 0]
    heights = anchors[:, :, 3] - anchors[:, :, 1]
    ctr_x = anchors[:, :, 0] + 0.5 * widths
    ctr_y = anchors[:, :, 1] + 0.5 * heights

    # Extract delta values
    dx = bbox_deltas[:, :, 0]
    dy = bbox_deltas[:, :, 1]
    dw = bbox_deltas[:, :, 2]
    dh = bbox_deltas[:, :, 3]

    # Apply deltas
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Convert center-size to corner coordinates
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    # Stack predictions
    proposals = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=2)

    return proposals