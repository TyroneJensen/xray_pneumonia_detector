import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import os
from PIL import Image
import matplotlib.pyplot as plt

# Custom Dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Model definition (using pre-trained ResNet18)
def create_model():
    model = models.resnet18(pretrained=True)
    # Modify the last layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def train_model(model, train_loader, val_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.3f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def prepare_data():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # You'll need to update these paths with your actual data paths
    data_dir = "chest_xray"
    train_normal = os.path.join(data_dir, "train", "NORMAL")
    train_pneumonia = os.path.join(data_dir, "train", "PNEUMONIA")
    
    # Create lists of image paths and labels
    train_images = []
    train_labels = []
    
    # Add paths and labels for training data
    for img in os.listdir(train_normal):
        if img.endswith('.jpeg'):
            train_images.append(os.path.join(train_normal, img))
            train_labels.append(0)  # 0 for normal
    
    for img in os.listdir(train_pneumonia):
        if img.endswith('.jpeg'):
            train_images.append(os.path.join(train_pneumonia, img))
            train_labels.append(1)  # 1 for pneumonia
    
    # Create datasets
    dataset = ChestXRayDataset(train_images, train_labels, transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Preparing data...")
    train_loader, val_loader = prepare_data()
    
    print("Creating model...")
    model = create_model()
    
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader)
    
    print("Training complete! Model saved as 'best_model.pth'")
