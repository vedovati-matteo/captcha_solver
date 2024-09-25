import os
from captcha_creation import Captcha
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools

def generate_random_string(length = None):
    """Generates a random string of the specified length.

    Args:
    length: The desired length of the string.

    Returns:
    A random string of the specified length.
    """
    
    #charset = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charset = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    if length is None:
        length = random.randint(4, 5)
    
    return ''.join(random.choice(charset) for i in range(length))

def image_generator():
    # generate 
    return Captcha(fonts=['fonts/cour.ttf']).generate_image(generate_random_string(5))

def save_image_and_annotation(index, output_dir, format="PASCAL_VOC_TXT"):
    """Generates an image and its annotation, and saves them to the disk."""
    image, boxes = image_generator()
    image_path = os.path.join(output_dir, f"{index}.jpg")
    label_path = os.path.join(output_dir, f"{index}.txt")

    # Save the image
    image.save(image_path)

    if format == "PASCAL_VOC_TXT":
        with open(label_path, 'w') as f:
            for box in boxes:
                class_char = box["char"]
                x_min, y_min, x_max, y_max = box["bb"]
                line = f"{class_char} {x_min} {y_min} {x_max} {y_max}\n"
                f.write(line)
    elif format == "YOLO":
        # Write the YOLO annotations
        with open(label_path, 'w') as f:
            for box in boxes:
                class_char = box["char"]
                x_min, y_min, x_max, y_max = box["bb"]

                # Clip bounding box coordinates to be within image dimensions
                x_min = max(0, min(x_min, image.size[0]))
                y_min = max(0, min(y_min, image.size[1]))
                x_max = max(0, min(x_max, image.size[0]))
                y_max = max(0, min(y_max, image.size[1]))
                
                # Convert bounding boxes to YOLO format
                width, height = image.size[1], image.size[0]
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                line = f"{class_char} {x_center} {y_center} {box_width} {box_height}\n"
                f.write(line)
    else:
        raise ValueError(f"Invalid format: {format}")

def create_dataset(output_dir, format="PASCAL_VOC_TXT", num_images=5000, max_workers=None):
    """Creates a dataset from an image generator using multiprocessing."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(functools.partial(save_image_and_annotation, output_dir=output_dir, format=format), range(num_images)), total=num_images))


if __name__ == "__main__":
    create_dataset("datasets/dataset_v6", num_images=200_000, max_workers=12)