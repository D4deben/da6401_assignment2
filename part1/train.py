import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
from model import OptimizedCNN

def get_dataloaders(data_dir, batch_size=256, val_split=0.2, augment=True):
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    targets = np.array(full_dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
    
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(datasets.ImageFolder(root=data_dir, transform=val_transforms), val_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, full_dataset.classes

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=20):
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += outputs.argmax(1).eq(labels).sum().item()
        
        val_acc = 100 * val_correct / len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', required=True)
    parser.add_argument('--wandb_entity', required=True)
    parser.add_argument('--data_dir', default='/kaggle/input/nature-12k/inaturalist_12K/train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--augment', action='store_true', default=True)
    
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=args.augment
    )

    model = OptimizedCNN(num_classes=len(classes))
    optimizer = optim.SGD(model.parameters(), 
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, args.epochs)
    wandb.finish()

if __name__ == '__main__':
    main()
