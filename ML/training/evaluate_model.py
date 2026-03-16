import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from training.train_model import get_data_loaders
from models.efficientnet_model import load_model

def evaluate_best_model(data_dir="dataset/gaussian_filtered_images", checkpoint_path="checkpoints/best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        
    print("Loading identical validation split...")
    # By using the same random_state=42 inside `get_data_loaders`, we recover the exact same validation 20%
    _, val_loader = get_data_loaders(data_dir, batch_size=32)
    
    model = load_model(checkpoint_path, device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nRunning inference over validation dataset...")
    # Inference loop
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/confusion_matrix.png')
    print("Saved confusion matrix heatmap to output/confusion_matrix.png")
    print("="*50)

if __name__ == "__main__":
    evaluate_best_model()
