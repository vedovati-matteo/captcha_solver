import torch
from PIL import Image
import os
from tqdm import tqdm

from bbox_detector.model import BBoxDetector
from bbox_detector.inference import inference as bbox_inference
from bbox_detector.train import load_checkpoint as bbox_load_checkpoint
from bbox_detector.utils import show_image

from char_recognizer.model import CharRecognizer
from char_recognizer.inference import inference as char_inference
from char_recognizer.inference import get_top_combinations
from char_recognizer.train import load_checkpoint as char_load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def solve_dataset_captcha(path, combination_number=3):
    bbox_model = BBoxDetector(dropout_rate=0.4, device=device).to(device)
    epoch, train_loss_history, val_loss_history, precision_recall_history = bbox_load_checkpoint('bbox_detector/checkpoints/model_epoch_40.pth', bbox_model, weights_only=False)

    char_model = CharRecognizer(dropout_rate=0.4, device=device).to(device)
    epoch, train_loss_history, val_loss_history = char_load_checkpoint('char_recognizer/checkpoints/model_epoch_35.pth', char_model, weights_only=False)
    
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    
    tot_images = len(image_files[:500])
    correct_images = 0
    
    for image_file in tqdm(image_files[:500], desc="Checking all the datatset images"):
        image_path = os.path.join(path, image_file)
        bb_path = os.path.join(path, os.path.splitext(image_file)[0] + '.txt')

        image = Image.open(image_path).convert("RGB")
    
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
    
        boxes, scores = bbox_inference(bbox_model, image, device, conf_threshold=0.95, nms_threshold=0.30)
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        
        chars_list = []
        for i in range(len(boxes)):
            chars = char_inference(char_model, image, sorted_boxes[i], device)
            chars_list.append(chars)
            
        top_combinations = get_top_combinations(chars_list, combination_number)
        top_combinations_str = [''.join([c for c, p in top_combinations[i][0]]) for i in range(len(top_combinations))]

        # check if the solution is correct
        if gt in top_combinations_str:
            correct_images += 1
    
    return correct_images, tot_images
    
if __name__ == "__main__":
    path = f'../datasets/dataset_v6'

    correct_images, tot_images = solve_dataset_captcha(path, combination_number=3)
    
    print(f"\n\nCorrect images: {correct_images}/{tot_images} = ({correct_images/tot_images*100:.2f}%)")