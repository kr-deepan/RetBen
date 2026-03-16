import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing.image_preprocessing import preprocess_image
from models.efficientnet_model import DRClassifier

class DRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Use our preprocessing block
        # Since preprocess_image returns a tensor ready for the model (with batch dim), 
        # but DataLoader expects individual samples, we handle it appropriately.
        # However, to be efficient with transforms, we'll implement standard loading here.
        import cv2
        from PIL import Image
        from preprocessing.image_preprocessing import circle_crop, apply_clahe
        
        img = cv2.imread(path)
        if img is None:
            # Fallback for broken images if any
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        cropped = circle_crop(img)
        enhanced = apply_clahe(cropped)
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            tensor = self.transform(img_pil)
        else:
            tensor = transforms.ToTensor()(img_pil)
            
        return tensor, torch.tensor(label, dtype=torch.long)

def get_data_loaders(data_dir, batch_size=32):
    # Mapping folders to class indices
    classes = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4}
    
    all_paths = []
    all_labels = []
    
    for class_name, class_idx in classes.items():
        folder_path = os.path.join(data_dir, class_name)
        if os.path.isdir(folder_path):
            images = glob.glob(os.path.join(folder_path, '*.png'))
            all_paths.extend(images)
            all_labels.extend([class_idx] * len(images))
            
    if not all_paths:
        raise ValueError(f"No images found in {data_dir}. Check dataset directory.")
        
    # Stratified split: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Calculate class weights for WeightedRandomSampler (handling imbalance)
    class_counts = [y_train.count(i) for i in range(5)]
    num_samples = len(y_train)
    class_weights = [num_samples / float(count) for count in class_counts]
    sample_weights = [class_weights[label] for label in y_train]
    
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights))
    
    # Augmentations for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Only resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DRDataset(X_train, y_train, transform=train_transform)
    val_dataset = DRDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_model(data_dir, epochs=15, batch_size=32, lr=1e-4, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training but CUDA is not available.")
        
    device = torch.device('cuda')
    print(f"Training on: {device}")
    
    print("Preparing dataloaders...")
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    model = DRClassifier(num_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
            
            train_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
        val_epoch_loss = val_loss / total_val
        val_epoch_acc = correct_val / total_val
        
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        
        # LR Scheduler step based on validation accuracy
        scheduler.step(val_epoch_acc)
        
        # Save Best Model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Acc: {best_val_acc:.4f} to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/gaussian_filtered_images', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    train_model(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
