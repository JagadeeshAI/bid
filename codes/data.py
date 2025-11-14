#!/usr/bin/env python3
# data.py (simplified - no remapping needed)
import os
from typing import Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_data_loaders_local(
    data_dir="/media/jag/volD2/cifer100/cifer",
    batch_size=64,
    num_workers=0,
    image_size=224,
    class_range: Tuple[int, int] = (0, 99),
    data_ratio: float = 1.0,
):
    """Load CIFAR-100 with original labels (0-99)"""
    
    start, end = class_range
    assert 0 <= start <= end <= 99

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    full_val = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform)

    valid_classes = list(range(start, end + 1))
    
    # Group indices by class for per-class sampling
    train_indices = []
    val_indices = []
    
    for class_id in valid_classes:
        # Get all indices for this specific class
        class_train_indices = [i for i, (_, l) in enumerate(full_train.samples) if l == class_id]
        class_val_indices = [i for i, (_, l) in enumerate(full_val.samples) if l == class_id]
        
        # Sample data_ratio percentage from each class
        if data_ratio < 1.0:
            num_train_samples = max(1, int(len(class_train_indices) * data_ratio))
            num_val_samples = max(1, int(len(class_val_indices) * data_ratio))
            class_train_indices = class_train_indices[:num_train_samples]
            class_val_indices = class_val_indices[:num_val_samples]
        
        train_indices.extend(class_train_indices)
        val_indices.extend(class_val_indices)

    train_loader = DataLoader(Subset(full_train, train_indices), batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(full_val, val_indices), batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader