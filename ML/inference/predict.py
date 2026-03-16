import torch
import torch.nn.functional as F
import json
from preprocessing.image_preprocessing import preprocess_image
from models.efficientnet_model import load_model

# Mapping class indices to severity labels
DR_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

def predict_image(image_path, model, device=None):
    """
    Runs the full inference pipeline (preprocessing -> model forward pass -> probabilities)
    Returns a dictionary structured for JSON response.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        # 1) Preprocess image to tensor
        tensor = preprocess_image(image_path, device)
        
        # 2) Run forward pass
        with torch.no_grad():
            logits = model(tensor)
            
        # 3) Compute probabilities using softmax
        probabilities = F.softmax(logits, dim=1).squeeze()
        
        # 4) Extract top class and confidence
        confidence, predicted_class_tensor = torch.max(probabilities, 0)
        
        predicted_class = predicted_class_tensor.item()
        confidence_score = confidence.item()
        severity_label = DR_CLASSES.get(predicted_class, "Unknown")
        
        # 5) Build output dictionary
        result = {
            "class": predicted_class,
            "severity": severity_label,
            "confidence": round(confidence_score, 4)
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to fundus image")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = load_model(args.checkpoint, device)
    
    print(f"Running inference on {args.image}...")
    result = predict_image(args.image, model, device)
    
    print("\nInference Result:")
    print(json.dumps(result, indent=2))
