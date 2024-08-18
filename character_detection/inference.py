import torch
from torchvision.ops import nms
from PIL import Image

def inference(model, image_path, device, transfrom, resize, conf_threshold, nms_threshold, top_k=5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    image = resize(image)
    
    image_tensor = transfrom(image).unsqueeze(0).to(device)

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