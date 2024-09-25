import torch
from torchvision.ops import nms
from PIL import Image

from .utils import transform, ResizeWithPad

def inference(model, image, device, conf_threshold, nms_threshold, top_k=5):
    # Load and preprocess the image
    resize = ResizeWithPad(224)
    image = resize(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        proposals, objectness_logits = model(image_tensor)

        # Post-process proposals
        scores = torch.softmax(objectness_logits[0], dim=1)[:, 1]
        boxes = proposals[0]

        # Filter by confidence threshold
        mask = scores > conf_threshold

        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        # Apply NMS
        keep = nms(filtered_boxes, filtered_scores, iou_threshold=nms_threshold)
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
    
    target = top_boxes.cpu().numpy().tolist()
    
    target_original = resize.revert_bounding_boxes(target)

    return target_original, top_scores.cpu().numpy().tolist()