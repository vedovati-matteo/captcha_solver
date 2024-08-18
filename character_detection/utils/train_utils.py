import torch
import torch.nn.functional as F

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