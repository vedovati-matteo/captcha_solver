import torch
import torch.nn.functional as F

from utils import *

def inference(model, image, bb, device, transform):
    cropped_img = image.crop(bb)
    
    # Apply other transformations
    cropped_img = transform(cropped_img)
    
    # Ensure the image is square (optional, but can help with some models)
    input = pad_to_square(cropped_img)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input.unsqueeze(0).to(device))
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        char = CharClass.get_char(predicted_class.item())
    
    return char