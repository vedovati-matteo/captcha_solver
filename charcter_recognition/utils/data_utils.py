import torch
from PIL import Image
import os
from PIL import ImageOps
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

class CharClass():
    #charset = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charset = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(self):
        pass
    
    def __len__(self):
        return len(self.charset)
    
    @staticmethod
    def get_char(index):
        return CharClass.charset[index]
    
    @staticmethod
    def get_index(char):
        return CharClass.charset.index(char)
        
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

def pad_to_square(image):
    c, h, w = image.shape
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(image, padding)

def custom_collate(batch):
    images, targets = zip(*batch)
    
    # Find the maximum dimensions in the batch
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    # Resize and pad images to the maximum size
    processed_images = []
    for img in images:
        c, h, w = img.shape
        
        # Calculate the scaling factor to fit the image within max_h and max_w
        scale = min(max_h / h, max_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize the image
        resized_img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        # Pad to match max_h and max_w
        padding = (0, max_w - new_w, 0, max_h - new_h)
        padded_img = F.pad(resized_img, padding)
        
        processed_images.append(padded_img)
    
    # Stack the processed images
    images = torch.stack(processed_images)
    targets = torch.tensor(targets)
    
    return images, targets

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform, max_crop_error=0.1):
        self.dataset_dir = dataset_dir
        
        # Get all image and annotation file paths
        self.image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        self.anno_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.txt')]
        
        # separate bounding boxes
        self.bounding_boxes = []
        for lemma in self.anno_paths:
            for i in range(5):
                self.bounding_boxes.append((lemma, i))
        
        self.transform = transform
        self.max_crop_error = max_crop_error
        
    def __len__(self):
        # Return the number of images
        return len(self.bounding_boxes)

    def add_crop_error(self, x_min, y_min, x_max, y_max, img_width, img_height):
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate maximum allowed error
        max_error_x = int(width * self.max_crop_error)
        max_error_y = int(height * self.max_crop_error)
        
        # Add random error to each coordinate
        x_min = max(0, x_min + random.randint(-max_error_x, max_error_x))
        y_min = max(0, y_min + random.randint(-max_error_y, max_error_y))
        x_max = min(img_width, x_max + random.randint(-max_error_x, max_error_x))
        y_max = min(img_height, y_max + random.randint(-max_error_y, max_error_y))
        
        return x_min, y_min, x_max, y_max
    
    def __getitem__(self, idx):
        # Return the image and the ground truth boxes
        image_path = self.image_paths[idx // 5]
        bounding_boxes_path, line_number = self.bounding_boxes[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        # Load annotations and get teh sepcific line
        if os.path.exists(bounding_boxes_path):
            with open(bounding_boxes_path, 'r') as f:
                for line_n, line in enumerate(f):
                    if line_number == line_n:
                        # (class_char, x_min, y_min, x_max, y_max)
                        data = line.strip().split(' ')
                        class_char = data[0] # not used
                        x_min, y_min, x_max, y_max = float(data[1]), float(data[2]), float(data[3]), float(data[4])
        
        x_min, y_min, x_max, y_max = self.add_crop_error(x_min, y_min, x_max, y_max, img_width, img_height)
        
        id_char = CharClass.get_index(class_char)
        
        cropped_img = image.crop((x_min, y_min, x_max, y_max))
            
        # Apply other transformations
        cropped_img = self.transform(cropped_img)
        
        # Ensure the image is square (optional, but can help with some models)
        cropped_img = pad_to_square(cropped_img)
        
        target = torch.tensor(id_char, dtype=torch.long)
        
        return cropped_img, target