# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import nms
from PIL import Image
import os
import numpy as np
from PIL import ImageOps
from tqdm import tqdm
from PIL.ImageDraw import Draw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_value = -1000.0
# %%

# %%
def generate_anchors(feature_map_size, stride):
    # Aspect ratios and scales for the anchor boxes
    # 0.3: 75, 90, 110
    # 0.4: 65, 80, 95
    # 0.5: 55, 70, 85
    # 0.6: 50, 65, 80
    # 0.7: 45, 60, 70
    
    ratios_scale = [[0.3, 75], [0.3, 90], [0.3, 110], 
                   [0.4, 65], [0.4, 80], [0.4, 95], 
                   [0.5, 55], [0.5, 70], [0.5, 85], 
                   [0.6, 50], [0.6, 65], [0.6, 80], 
                   [0.7, 45], [0.7, 60], [0.7, 70]]
    
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

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the final layers
        
        # Freeze the ResNet50 layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1)  # Adjust input channels to 2048
        self.cls_logits = nn.Conv2d(512, 15 * 2, kernel_size=1, stride=1)  # 12 anchors, 2 classes (object/background)
        self.bbox_pred = nn.Conv2d(512, 15 * 4, kernel_size=1, stride=1)  # 12 anchors, 4 bbox coordinates
    
    def forward(self, x): # [batch_size, 3, 224, 224]
        batch_size, _, _, _ = x.shape
        
        # Pass through the backbone
        features = self.backbone(x)
        conv_features = F.relu(self.conv(features))        
        objectness_logits = self.cls_logits(conv_features)  # Predict the probability of an object being present in an anchor box (Shape: [batch_size, 24, H, W])
        # Two classes: object (1) and background (0)
        bbox_deltas = self.bbox_pred(conv_features)         # Refine the anchor boxes to better match the ground truth bounding boxes (Shape: [batch_size, 48, H, W])
        # Four values: (dx, dy, dw, dh)
        
        # Generate anchor boxes
        feature_map_size = conv_features.shape[-2:]  # (H, W) => 7, 7
        stride = x.shape[-2] // feature_map_size[0]  # Assuming square input and feature map (doing it only in one axis) => 224 / 7 = 32
        
        anchors = generate_anchors(feature_map_size, stride) # [num_anchors (735), 4]
        
        # Reshape objectness_logits [batch_size, num_anchors, 2]
        objectness_logits = objectness_logits.permute(0, 2, 3, 1).contiguous()
        objectness_logits = objectness_logits.view(batch_size, -1, 2)
        
        # Proposals
        proposals = apply_deltas_to_anchors(anchors, bbox_deltas) # [batch_size, num_anchors, 4]
        
        # Return the necessary outputs for loss computation
        # proposals: [batch_size, num_anchors, 4], objectness_logits: [batch_size, num_anchors, 2]
        return proposals, objectness_logits
    
# %%

def box_iou(box1, box2):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box1 (Tensor[N, 4]): first set of boxes
        box2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def compute_rpn_loss(proposals, targets, objectness_logits, lambda_=10.0):
    # proposals: [batch_size, num_anchors, 4]
    # targets: [batch_size, 5, 4]
    # objectness_logits: [batch_size, num_anchors, 2]
    
    batch_size = proposals.shape[0]
    
    total_classification_loss = 0
    total_bbox_regression_loss = 0
    total_positive_samples = 0

    for i in range(batch_size):
        # Process each image in the batch separately
        image_proposals = proposals[i]  # [num_anchors, 4]
        image_targets = targets[i]  # [5, 4]
        # remove padding
        mask = (image_targets != -1000.0).any(dim=1)
        image_targets = image_targets[mask] # [4 or 5, 4]
        
        image_objectness_logits = objectness_logits[i]  # [num_anchors, 2]

        # Compute IoU between proposals and targets for this image
        ious = box_iou(image_proposals, image_targets)  # [num_anchors, 5 or 4]

        # Assign labels
        objectness_targets = torch.zeros(image_proposals.shape[0], dtype=torch.long, device=proposals.device)
        
        positive_indices = (ious > 0.3).any(dim=1)
        objectness_targets[positive_indices] = 1
        
        max_ious, _ = ious.max(dim=1)
        negative_indices = (max_ious < 0.3)
        objectness_targets[negative_indices] = 0
        
        ignore_indices = (~positive_indices) & (~negative_indices)
        objectness_targets[ignore_indices] = -1

        # Add class balancing
        num_pos = positive_indices.sum().float()
        num_neg = negative_indices.sum().float()
        pos_weight = num_neg / (num_pos + 1e-8)
        
        classification_loss = F.cross_entropy(
            image_objectness_logits,
            objectness_targets,
            ignore_index=-1,
            reduction='none'
        ).sum()
        #classification_loss = (classification_loss * pos_weight).sum()

        # Bounding box regression loss for this image
        if positive_indices.sum() > 0: # if there are positive samples
            positive_proposals = image_proposals[positive_indices]
            positive_targets = image_targets[torch.argmax(ious[positive_indices], dim=1)]
            
            bbox_regression_loss = F.smooth_l1_loss(
                positive_proposals,
                positive_targets,
                reduction='sum'
            )
        else:
            bbox_regression_loss = torch.tensor(0.0, device=proposals.device)

        total_classification_loss += classification_loss
        total_bbox_regression_loss += bbox_regression_loss
        total_positive_samples += positive_indices.sum()

    # Normalize losses
    avg_classification_loss = total_classification_loss / batch_size
    avg_bbox_regression_loss = total_bbox_regression_loss / (total_positive_samples + 1e-8)
    
    # Total loss
    loss = avg_classification_loss + lambda_ * avg_bbox_regression_loss

    return loss, avg_classification_loss, avg_bbox_regression_loss

# %%
class ResizeWithPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        delta_width = self.target_size - width
        delta_height = self.target_size - height
        
        pad_width = delta_width // 2 + delta_width % 2
        pad_height = delta_height // 2 + delta_height % 2

        self.padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height) # (left, top, right, bottom)
        
        image = ImageOps.expand(image, border=self.padding, fill=0)
                
        return image
    
    def adjust_bounding_box(self, bounding_boxes):
        """Adjusts bounding box coordinates after padding.

        Args:
            bounding_boxes: A list containing the bounding boxes, each bouning box is a tuple like so (x_min, y_min, x_max, y_max).

        Returns:
            A list containing the adjusted bounding boxes.
        """
        
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        
        adjusted_bounding_boxes = []
        
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            new_x_min = x_min + pad_left
            new_y_min = y_min + pad_top
            new_x_max = x_max + pad_left
            new_y_max = y_max + pad_top
            adjusted_bounding_boxes.append((new_x_min, new_y_min, new_x_max, new_y_max))
        
        return adjusted_bounding_boxes

class RPNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform, resize = None):
        self.dataset_dir = dataset_dir
        
        # Get all image and annotation file paths
        self.image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        self.anno_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.txt')]
        
        self.resize_with_pad = None
        if resize is not None:
            self.resize_with_pad = resize
        
        self.transform = transform

    def __len__(self):
        # Return the number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Return the image and the ground truth boxes
        image_path = self.image_paths[idx]
        anno_path = self.anno_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations (if annotation file exists)
        target = []
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    # Parse YOLO annotation format (class_char, x_min, y_min, x_max, y_max)
                    data = line.strip().split(' ')
                    class_char = data[0] # not used
                    x_min, y_min, x_max, y_max = float(data[1]), float(data[2]), float(data[3]), float(data[4])
                    target.append((x_min, y_min, x_max, y_max))
        
        # Apply resizing and padding if necessary
        if self.resize_with_pad is not None: 
            image = self.resize_with_pad(image)
            
            target = self.resize_with_pad.adjust_bounding_box(target)
            
        # Apply other transformations
        image = self.transform(image)
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return image, target

def custom_collate_fn(batch):
    images = []
    targets = []
    
    for b in batch:
        images.append(b[0])
        targets.append(b[1])
    
    images = torch.stack(images, dim=0)
    
    # Pad targets to the same length
    max_len = max([t.size(0) for t in targets])
    padded_targets = torch.ones((len(targets), max_len, 4)) * pad_value # pad as -1000.0 so to not have problems with calcualtions
    
    for i, t in enumerate(targets):
        padded_targets[i, :t.size(0), :] = t
    
    return images, padded_targets

# %%

def calculate_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Calculate Average Precision for a single image.
    
    Args:
    pred_boxes (Tensor): Predicted bounding boxes (N, 4)
    pred_scores (Tensor): Predicted confidence scores (N,)
    gt_boxes (Tensor): Ground truth bounding boxes (M, 4)
    iou_threshold (float): IoU threshold for considering a positive detection
    
    Returns:
    float: Average Precision
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0  # Perfect score if no objects and no predictions
    if len(gt_boxes) == 0:
        return 0.0  # All predictions are false positives
    if len(pred_boxes) == 0:
        return 0.0  # All ground truths are missed
    
    # Sort predictions by score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Compute IoU between pred and gt boxes
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    
    # Initialize variables
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
    
    # Match predictions to ground truth
    for i in range(len(pred_boxes)):
        max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            gt_matched[max_idx] = True
            pred_matched[i] = True
    
    # Compute precision and recall at each prediction
    true_positives = torch.cumsum(pred_matched, dim=0)
    false_positives = torch.cumsum(~pred_matched, dim=0)
    false_negatives = torch.sum(~gt_matched)
    
    precisions = true_positives / (true_positives + false_positives)
    recalls = true_positives / len(gt_boxes)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if torch.sum(recalls >= t) == 0:
            p = 0
        else:
            p = torch.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap.item()

def calculate_precision_recall_f1(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate Precision, Recall, and F1-score for a single image.
    
    Args:
    pred_boxes (Tensor): Predicted bounding boxes (N, 4)
    pred_scores (Tensor): Predicted confidence scores (N,)
    gt_boxes (Tensor): Ground truth bounding boxes (M, 4)
    iou_threshold (float): IoU threshold for considering a positive detection
    score_threshold (float): Confidence score threshold for considering a detection
    
    Returns:
    tuple: (Precision, Recall, F1-score)
    """
    # Filter predictions by score threshold
    mask = pred_scores >= score_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0  # Perfect score if no objects and no predictions
    if len(gt_boxes) == 0:
        return 0.0, 1.0, 0.0  # All predictions are false positives
    if len(pred_boxes) == 0:
        return 1.0, 0.0, 0.0  # All ground truths are missed
    
    # Compute IoU between pred and gt boxes
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    
    # Match predictions to ground truth
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
    
    for i in range(len(pred_boxes)):
        max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            gt_matched[max_idx] = True
            pred_matched[i] = True
    
    true_positives = torch.sum(pred_matched)
    false_positives = torch.sum(~pred_matched)
    false_negatives = torch.sum(~gt_matched)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision.item(), recall.item(), f1_score.item()

# %%

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset = RPNDataset('datasets/dataset_v2', transform=transform, resize=ResizeWithPad(224))

total_size = len(dataset)
train_size = int(0.80 * total_size)
val_size = int(0.10 * total_size)
test_size = total_size - train_size - val_size

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Set the batch size
batch_size = 128

# Create the DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Create the DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
# %%
def show_rand_image(train_loader):
    extracted = next(iter(train_loader))
    image, target = extracted
    image = image[0]
    target = target[0]
    target = target.tolist()
    
    to_pil = transforms.ToPILImage()
    image = to_pil(image)
    
    draw = Draw(image)
    for bb in target:
        draw.rectangle(bb, outline=(255, 0, 0), width=2)
    image.show()


# %%
num_epochs = 5

# Assuming you have a DataLoader named 'train_loader' and ground truth data
model = RPN()
# Move model to device
model = model.to(device)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        proposals, objectness_logits = model(images)
        
        # Compute loss here
        loss, classification_loss, bbox_regression_loss = compute_rpn_loss(proposals, targets, objectness_logits)
        
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    
    # Evaluation loop
    model.eval()
    total_val_loss = 0.0
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            proposals, objectness_logits = model(images)
            
            # Compute validation loss
            val_loss, _, _ = compute_rpn_loss(proposals, targets, objectness_logits)
            total_val_loss += val_loss.item()
            
            # Post-process proposals
            batch_size = images.size(0)
            for i in range(batch_size):
                scores = torch.softmax(objectness_logits[i], dim=1)[:, 1]
                boxes = proposals[i]
                
                image_targets = targets[i]
                # remove padding
                mask = (image_targets != -1000.0).any(dim=1)
                image_targets = image_targets[mask] # [4 or 5, 4]
                
                # Apply NMS
                keep = nms(boxes, scores, iou_threshold=0.3)
                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]
                
                # Keep top-k proposals k=10
                k = min(10, filtered_boxes.size(0))
                top_scores, top_idx = filtered_scores.topk(k)
                top_boxes = filtered_boxes[top_idx]
                
                all_pred_boxes.append(top_boxes.cpu())
                all_pred_scores.append(top_scores.cpu())
                all_gt_boxes.append(image_targets.cpu())
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Calculate mAP
    ap_sum = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        ap = calculate_ap(pred_boxes, pred_scores, gt_boxes)
        precision, recall, f1 = calculate_precision_recall_f1(pred_boxes, pred_scores, gt_boxes)
        
        ap_sum += ap
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    num_images = len(all_pred_boxes)
    mAP = ap_sum / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_f1 = total_f1 / num_images
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, mAP: {mAP:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-score: {avg_f1:.4f}")

# %%

def inference(model, image_path, device, transfrom, resize, conf_threshold=0.7, nms_threshold=0.3, top_k=10):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    image = resize(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        proposals, objectness_logits = model(image_tensor)
        print("Raw proposals shape:", proposals.shape)
        print("Objectness logits shape:", objectness_logits.shape)
        print("Max objectness score:", objectness_logits.max().item())
        print("Min objectness score:", objectness_logits.min().item())

        # Post-process proposals
        scores = torch.softmax(objectness_logits[0], dim=1)[:, 1]
        print("After softmax, max score:", scores.max().item())
        boxes = proposals[0]

        # Filter by confidence threshold
        mask = scores > conf_threshold
        print("Number of proposals after confidence thresholding:", mask.sum().item())
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        print("Filtered boxes shape:", filtered_boxes.shape)
        print("Filtered scores shape:", filtered_scores.shape)

        # Apply NMS
        keep = nms(filtered_boxes, filtered_scores, iou_threshold=nms_threshold)
        print("Number of boxes after NMS:", len(keep))
        nms_boxes = filtered_boxes[keep]
        nms_scores = filtered_scores[keep]

        # Keep top-k proposals
        k = min(top_k, nms_boxes.size(0))
        top_scores, top_idx = nms_scores.topk(k)
        top_boxes = nms_boxes[top_idx]

    # Convert boxes to original image scale
    orig_w, orig_h = image.size
    scale_w = orig_w / 224  # Adjust if using a different resize dimension
    scale_h = orig_h / 224
    top_boxes[:, [0, 2]] *= scale_w
    top_boxes[:, [1, 3]] *= scale_h

    return top_boxes.cpu().numpy(), top_scores.cpu().numpy()

# %%

image_path = 'datasets/dataset_v2/1.jpg'
boxes, scores = inference(model, image_path, device, transform, ResizeWithPad(224))
# %%
image = Image.open(image_path).convert("RGB")
image = ResizeWithPad(224)(image)

target = boxes.tolist()


draw = Draw(image)
for bb in target:
    draw.rectangle(bb, outline=(255, 0, 0), width=2)
image.show()
# %%
