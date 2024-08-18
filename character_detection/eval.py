import torch
from torchvision.ops import nms
from tqdm import tqdm

from utils.train_utils import compute_loss
from utils.eval_utils import calculate_ap, calculate_precision_recall_f1

def evaluate(model, val_loader, conf_threshold, nms_threshold, device, top_k=5):
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
            val_loss, _, _ = compute_loss(proposals, targets, objectness_logits)
            total_val_loss += val_loss.item()
            
            # Post-process proposals
            batch_size = images.size(0)
            for i in range(batch_size):
                scores = torch.softmax(objectness_logits[i], dim=1)[:, 1]
                boxes = proposals[i]
                
                image_targets = targets[i]
                
                mask = scores > conf_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
                # remove padding
                #mask = (image_targets != -1000.0).any(dim=1)
                #image_targets = image_targets[mask] # [4 or 5, 4]
                
                # Apply NMS
                keep = nms(filtered_boxes, filtered_scores, iou_threshold=nms_threshold)
                nms_boxes = filtered_boxes[keep]
                nms_scores = filtered_scores[keep]
                
                # Keep top-k proposals k=10
                k = min(top_k, nms_boxes.size(0))
                top_scores, top_idx = nms_scores.topk(k)
                top_boxes = nms_boxes[top_idx]
                
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
    
    return avg_val_loss, mAP, avg_precision, avg_recall, avg_f1
    
    