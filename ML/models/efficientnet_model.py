import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(DRClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the final classification layer
        # The original classifier is a Sequential block with Dropout(p=0.2) and Linear(in_features=1280, out_features=1000)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Softmax is typically applied at the inference stage since 
        # CrossEntropyLoss expects raw logits during training.
        
    def forward(self, x):
        # Returns raw logits
        return self.backbone(x)

def load_model(checkpoint_path, device=None, num_classes=5):
    """
    Utility to load the DRClassifier from a saved PyTorch checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = DRClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Test the model structure and GPU capability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRClassifier().to(device)
    
    print(f"Model loaded on {device}.")
    print("Classifier head:")
    print(model.backbone.classifier)
    
    # Dummy pass to verify functionality
    test_tensor = torch.randn(1, 3, 224, 224).to(device)
    out = model(test_tensor)
    print(f"Output shape: {out.shape}")
