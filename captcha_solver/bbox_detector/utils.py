from PIL import Image
from PIL.ImageDraw import Draw
from matplotlib import pyplot as plt
import torch
from PIL import Image
import os
from PIL import ImageOps
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def show_image(image, target_bb):

    image_copy = image.copy()

    draw = Draw(image_copy)
    for bb in target_bb:
        draw.rectangle(bb, outline=(255, 0, 0), width=2)
    image_copy.show() 
    
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

pad_value=-1000.0

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

transform_noise = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResizeWithPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.original_size = None
        self.padding = None

    def __call__(self, image):
        self.original_size = image.size
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
    
    def revert_bounding_boxes(self, bounding_boxes):
        """Reverts bounding box coordinates to the original image size and position.

        Args:
            bounding_boxes: A list containing the bounding boxes, each bounding box is a list like so [x_min, y_min, x_max, y_max].

        Returns:
            A list containing the reverted bounding boxes.
        """
        if self.original_size is None or self.padding is None:
            raise ValueError("The image must be processed before reverting bounding boxes.")

        pad_left, pad_top, pad_right, pad_bottom = self.padding
        original_width, original_height = self.original_size
        scale_x = original_width / (self.target_size - pad_left - pad_right)
        scale_y = original_height / (self.target_size - pad_top - pad_bottom)

        reverted_bounding_boxes = []

        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            new_x_min = max(0, (x_min - pad_left) * scale_x)
            new_y_min = max(0, (y_min - pad_top) * scale_y)
            new_x_max = min(original_width, (x_max - pad_left) * scale_x)
            new_y_max = min(original_height, (y_max - pad_top) * scale_y)
            reverted_bounding_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

        return reverted_bounding_boxes

class BBoxDetectorDataset(torch.utils.data.Dataset):
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

def collate_fn(batch):
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

def compute_loss(proposals, targets, objectness_logits, lambda_=10.0):
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