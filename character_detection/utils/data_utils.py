import torch
from PIL import Image
import os
from PIL import ImageOps
from torchvision import transforms

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