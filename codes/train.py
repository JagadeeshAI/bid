# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from data import get_data_loaders_local
from utils import get_model, evaluate
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-100')
    parser.add_argument('--range', type=str, default='0-49', 
                       help='Class range to train on (e.g., 0-49, 10-59)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lora-rank', type=int, default=0,
                       help='LoRA rank (0 for full fine-tuning)')
    parser.add_argument('--data-ratio', type=float, default=1.0,
                       help='Fraction of data to use for training')
    return parser.parse_args()

def parse_range(range_str):
    """Parse class range string like '0-49' into tuple (0, 49)"""
    try:
        start, end = range_str.split('-')
        return (int(start), int(end))
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Expected format: 'start-end' (e.g., '0-49')")

def main():
    args = parse_args()
    
    # Parse class range
    class_range = parse_range(args.range)
    start_class, end_class = class_range
    
    # Config
    num_epochs = args.epochs
    learning_rate = args.lr
    num_classes = 100
    lora_rank = args.lora_rank
    data_ratio = args.data_ratio
    
    # Save directory and checkpoint naming
    save_dir = 'checkpoints/oracle'
    checkpoint_name = f'best_{start_class}_{end_class}.pth'
    if lora_rank > 0:
        checkpoint_name = f'best_{start_class}_{end_class}_lora{lora_rank}.pth'
    
    os.makedirs(save_dir, exist_ok=True)
    print(f'🚀 Training for {num_epochs} epochs | Device: {DEVICE}')
    print(f'📊 Class Range: {start_class}-{end_class} | Data Ratio: {data_ratio:.2f}')
    print(f'🔧 LoRA Rank: {lora_rank} | Learning Rate: {learning_rate}\n')
    
    # Data
    train_loader, val_loader = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=data_ratio, class_range=class_range
    )
    print(f'📦 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}\n')
    
    # Model
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        lora_rank=lora_rank,
        device=DEVICE
    )
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f'✅ Best model saved: {checkpoint_path} | Val Acc: {val_acc:.2f}%')
    
    print(f'\n{"="*60}')
    print(f'🎉 Training completed! Best Val Acc: {best_val_acc:.2f}%')
    print(f'💾 Final model saved as: {checkpoint_name}')
    print(f'📁 Save directory: {save_dir}')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()