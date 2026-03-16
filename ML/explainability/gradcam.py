import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

from models.efficientnet_model import load_model
from preprocessing.image_preprocessing import preprocess_image, circle_crop, apply_clahe

def generate_gradcam(image_path, model, target_class=None, output_dir="output", device=None):
    """
    Generates a GradCAM heatmap for a given image and model.
    Saves the overlaid heatmap to the output directory.
    Returns the path to the saved heatmap.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess the image to tensor for the model
    input_tensor = preprocess_image(image_path, device)
    # GradCAM requires gradients to compute feature importance
    input_tensor.requires_grad_(True)
    
    # 2. Extract the original image for overlaying
    # We re-run crop+clahe to get the exact base image the model sees before normalization
    raw_img = cv2.imread(image_path)
    cropped = circle_crop(raw_img)
    enhanced = apply_clahe(cropped)
    
    # Needs to match the model input size (224x224)
    enhanced_resized = cv2.resize(enhanced, (224, 224))
    
    # Convert BGR to RGB for PIL
    rgb_img = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # 3. Hook GradCAM into the last convolutional layer of EfficientNet-B0
    # For torchvision efficientnet_b0, features[-1] is a list/Sequential of layers. The Conv2d is at [0].
    cam_extractor = GradCAM(model, target_layer=model.backbone.features[-1][0])
    
    # 4. Forward pass and CAM extraction need gradients
    # Temporarily force all parameters to require gradients just in case it dropped
    for param in model.parameters():
        param.requires_grad_(True)
        
    model.train()
    with torch.set_grad_enabled(True):
        out = model(input_tensor)
        
        # Determine target class (use predicted if none provided)
        if target_class is None:
            target_class = out.squeeze(0).argmax().item()
            
        # 5. Retrieve the CAM
        # GradCAM requires the class index and the model output scores
        cams = cam_extractor(class_idx=target_class, scores=out)

    # Revert model to eval mode
    model.eval()
    
    # We only have one target class, get the first cam
    cam = cams[0]
    
    # 6. Overlay the CAM on the original image
    # The overlay_mask utility takes a PIL Image and a binary mask/tensor
    overlay = overlay_mask(pil_img, to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
    
    # Save the result
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"heatmap_{filename}")
    overlay.save(output_path)
    
    # Clean up hooks
    cam_extractor.clear_hooks()
    
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to fundus image")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = load_model(args.checkpoint, device)
    
    heatmap_path = generate_gradcam(args.image, model, device=device)
    print(f"Heatmap saved to: {heatmap_path}")
