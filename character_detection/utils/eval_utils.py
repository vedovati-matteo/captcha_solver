import torch
import numpy as np
from .train_utils import box_iou

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
