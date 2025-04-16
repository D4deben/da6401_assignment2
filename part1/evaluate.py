import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import defaultdict
import wandb
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import OptimizedCNN

def create_prediction_grid(images, labels, preds, class_names, n_rows=10, n_cols=3, title="Predictions"):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*2))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(images):
                break
                
            ax = axes[i,j]
            img = images[idx].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip((img * std + mean), 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            
            true_label = class_names[labels[idx]]
            pred_label = class_names[preds[idx]]
            is_correct = preds[idx] == labels[idx]
            
            title_color = 'green' if is_correct else 'red'
            title_text = f"True: {true_label}\nPred: {pred_label}"
            ax.set_title(title_text, fontsize=9, color=title_color, pad=2)
    
    plt.tight_layout()
    return fig

def evaluate_testset(model_path, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    class_names = test_dataset.classes

    model = OptimizedCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")

    wandb.init(project="DL_A2", name="test_evaluation")
    wandb.log({"test_accuracy": accuracy})

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    accuracy_table = wandb.Table(columns=["Class", "Accuracy", "Samples"])
    for class_idx in range(len(class_names)):
        if class_total[class_idx] > 0:
            acc = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            acc = float('nan')
        accuracy_table.add_data(class_names[class_idx], acc, class_total[class_idx])
    
    wandb.log({"class_accuracy": accuracy_table})

    present_classes = [c for c in range(len(class_names)) if class_total[c] > 0]
    if present_classes:
        cm = confusion_matrix(all_labels, all_preds, labels=present_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[class_names[c] for c in present_classes],
                   yticklabels=[class_names[c] for c in present_classes])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

    num_samples = 30
    if len(all_images) >= num_samples:
        indices = random.sample(range(len(all_images)), num_samples)
        sample_images = [all_images[i] for i in indices]
        sample_labels = [all_labels[i] for i in indices]
        sample_preds = [all_preds[i] for i in indices]
        
        grid_fig = create_prediction_grid(
            sample_images, sample_labels, sample_preds, 
            class_names, title="Test Set Predictions"
        )
        wandb.log({"prediction_grid": wandb.Image(grid_fig)})
        plt.close(grid_fig)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--wandb_project', default='DL_A2_eval')
    parser.add_argument('--wandb_entity', required=True)
    args = parser.parse_args()
    
    evaluate_testset(args.model_path, args.test_dir)
