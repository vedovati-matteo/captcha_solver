import torch
import torch.nn.functional as F
from itertools import product

from .utils import *

def inference(model, image, bb, device, top_k=5):
    cropped_img = image.crop(bb)
    
    # Apply other transformations
    cropped_img = transform(cropped_img)
    
    # Ensure the image is square (optional, but can help with some models)
    input = pad_to_square(cropped_img)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input.unsqueeze(0).to(device))
        probabilities = F.softmax(outputs, dim=1)
        # Get top k probabilities and their indices
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        # Convert to list of (char, prob) tuples
        char_probs = [(CharClass.get_char(idx.item()), prob.item()) 
                      for idx, prob in zip(top_indices[0], top_probs[0])]
    
    return char_probs

def get_top_combinations(chars_list, num_shots):
    # Generate all possible combinations
    all_combinations = list(product(*chars_list))
    
    # Calculate probability for each combination
    combo_probs = []
    for combo in all_combinations:
        prob = 1
        for char, char_prob in combo:
            prob *= char_prob
        combo_probs.append((combo, prob))
    
    # Sort combinations by probability in descending order
    sorted_combos = sorted(combo_probs, key=lambda x: x[1], reverse=True)
    
    # Return top num_shots combinations
    return sorted_combos[:num_shots]