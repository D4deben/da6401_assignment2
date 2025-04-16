import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import wandb
from model import OptimizedCNN

def visualize_first_layer(model_path, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    random_idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[random_idx]
    img = img.unsqueeze(0).to(device)
    
    model = OptimizedCNN(num_classes=len(test_dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    first_conv = model.conv_blocks[0]
    num_filters = first_conv.out_channels
    filters = first_conv.weight.data.cpu().numpy()
    
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    plt.figure(figsize=(12, 12))
    for i in range(num_filters):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(filters[i, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle(f'First Layer Filters ({num_filters} total)', fontsize=16)
    plt.tight_layout()
    filters_fig = plt.gcf()
    
    model.eval()
    with torch.no_grad():
        feature_maps = first_conv(img)
    
    fmaps = feature_maps.squeeze(0).cpu().numpy()
    fmap_min, fmap_max = fmaps.min(), fmaps.max()
    fmaps = (fmaps - fmap_min) / (fmap_max - fmap_min)
    
    plt.figure(figsize=(12, 12))
    for i in range(num_filters):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(fmaps[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps ({num_filters} total)', fontsize=16)
    plt.tight_layout()
    fmap_fig = plt.gcf()
    
    img_denorm = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_denorm = img_denorm * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_denorm = np.clip(img_denorm, 0, 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_denorm)
    plt.title(f'Original Test Image\nClass: {test_dataset.classes[label]}')
    plt.axis('off')
    orig_fig = plt.gcf()
    
    filter_magnitudes = torch.norm(first_conv.weight.data.view(num_filters, -1), p=2, dim=1).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.hist(filter_magnitudes, bins=20)
    plt.title('Filter Magnitude Distribution')
    plt.xlabel('Magnitude (L2 norm)')
    plt.ylabel('Count')
    magnitude_fig = plt.gcf()
    
    activation_means = feature_maps.mean(dim=(0, 2, 3)).cpu().numpy()
    activation_max = feature_maps.amax(dim=(0, 2, 3)).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_filters), activation_means, alpha=0.5, label='Mean')
    plt.bar(range(num_filters), activation_max, alpha=0.5, label='Max')
    plt.title('Feature Map Activation Statistics')
    plt.xlabel('Filter Index')
    plt.ylabel('Activation Value')
    plt.legend()
    activation_fig = plt.gcf()
    
    wandb.init(project="DL_A2", name="filter_visualization")
    wandb.log({
        "original_image": wandb.Image(orig_fig),
        "first_layer_filters": wandb.Image(filters_fig),
        "feature_maps": wandb.Image(fmap_fig),
        "filter_magnitudes": wandb.Image(magnitude_fig),
        "activation_stats": wandb.Image(activation_fig),
        "selected_class": test_dataset.classes[label]
    })
    
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--wandb_project', default='DL_A2_visualize')
    parser.add_argument('--wandb_entity', required=True)
    args = parser.parse_args()
    
    visualize_first_layer(args.model_path, args.test_dir)
