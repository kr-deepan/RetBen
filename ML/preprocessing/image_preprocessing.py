import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def circle_crop(img):
    """
    Crops the black borders from a retinal fundus image based on its circular shape.
    Expects a NumPy BGR image (OpenCV format).
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask using a threshold to find the retina area
    # Background is black, retina is brighter
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img  # Return original if no contour found
        
    # Get the largest contour assuming it's the fundus
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image to this bounding box
    cropped = img[y:y+h, x:x+w]
    return cropped

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the L-channel of the LAB color space.
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab_eq = cv2.merge((l_eq, a, b))
    img_bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    return img_bgr_eq

def resize_normalize(img_cv2, target_size=(224, 224)):
    """
    Resizes the image and normalizes it using ImageNet weights.
    Returns a PyTorch tensor.
    """
    # Convert completely to PIL Image from OpenCV BGR format
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img_pil)
    return tensor

def preprocess_image(image_path, device=None):
    """
    Full pipeline: reads image -> crops -> CLAHE -> resize/normalize -> GPU tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
        
    # Apply processing steps
    cropped = circle_crop(img)
    enhanced = apply_clahe(cropped)
    tensor = resize_normalize(enhanced)
    
    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor

if __name__ == '__main__':
    # Test block
    print("Preprocessing module ready.")
