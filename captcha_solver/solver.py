import torch
from PIL import Image

from bbox_detector.model import BBoxDetector
from bbox_detector.inference import inference as bbox_inference
from bbox_detector.train import load_checkpoint as bbox_load_checkpoint
from bbox_detector.utils import show_image

from char_recognizer.model import CharRecognizer
from char_recognizer.inference import inference as char_inference
from char_recognizer.inference import get_top_combinations
from char_recognizer.train import load_checkpoint as char_load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def solve_captcha(image, combination_number=3):
    bbox_model = BBoxDetector(dropout_rate=0.4, device=device).to(device)
    epoch, train_loss_history, val_loss_history, precision_recall_history = bbox_load_checkpoint('bbox_detector/checkpoints/model_epoch_40.pth', bbox_model, weights_only=False)

    char_model = CharRecognizer(dropout_rate=0.4, device=device).to(device)
    epoch, train_loss_history, val_loss_history = char_load_checkpoint('char_recognizer/checkpoints/model_epoch_35.pth', char_model, weights_only=False)
    
    boxes, scores = bbox_inference(bbox_model, image, device, conf_threshold=0.95, nms_threshold=0.30)
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    
    chars_list = []
    for i in range(len(boxes)):
        chars = char_inference(char_model, image, sorted_boxes[i], device)
        chars_list.append(chars)
        
    top_combinations = get_top_combinations(chars_list, combination_number)
    
    for i, combination in enumerate(top_combinations):
        print(f"Combination {i + 1}: {''.join([c for c, p in top_combinations[i][0]])}, probability: {top_combinations[i][1]}")
    
    return [''.join([c for c, p in top_combinations[i][0]]) for i in range(len(top_combinations))]

if __name__ == "__main__":
    import sys
    
    image_number = sys.argv[1]
    
    image_path = f'../datasets/dataset_v6/{image_number}.jpg'
    image = Image.open(image_path).convert("RGB")
    
    bb_path = f'../datasets/dataset_v6/{image_number}.txt'
    bbs = []
    c_s = []
    with open(bb_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(' ')
            class_char = data[0] # not used
            x_min, y_min, x_max, y_max = float(data[1]), float(data[2]), float(data[3]), float(data[4])
            c_s.append(class_char)
            bbs.append((x_min, y_min, x_max, y_max))
    
    gt = ''.join(c_s)
    
    captcha_solution = solve_captcha(image)
    
    print("\n\n")
    
    # check if the solution is correct
    if gt in captcha_solution:
        print("✓ Captcha solved correctly!")
    else:
        print("X Captcha not solved")
    
    print("Ground truth:", gt)
    print("Captcha solution:", captcha_solution)