#!/usr/bin/env python3
# eval.py
import torch
from torch import nn
from data import get_data_loaders_local
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE
from utils import get_model, evaluate

def main():
    print("📊 Evaluating Model on Different Ranges\n")
    
    # Load model
    model = get_model(
        num_classes=100,
        pretrained=False,
        lora_rank=0,
        checkpoint_path="checkpoints/best_0_49.pth",
        device=DEVICE
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on 0-9 (should be high)
    print("\n" + "="*60)
    print("📊 Classes 0-9:")
    _, loader_0_9 = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=1.0, class_range=(0, 9)
    )
    loss_0_9, acc_0_9 = evaluate(model, loader_0_9, criterion, DEVICE)
    print(f"   Accuracy: {acc_0_9 * 100:.2f}%")
    
    # Evaluate on 10-49 (should be high)
    print("\n" + "="*60)
    print("📊 Classes 10-49:")
    _, loader_10_49 = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=1.0, class_range=(10, 49)
    )
    loss_10_49, acc_10_49 = evaluate(model, loader_10_49, criterion, DEVICE)
    print(f"   Accuracy: {acc_10_49 * 100:.2f}%")
    
    # Evaluate on 50-59 (should be low/random)
    print("\n" + "="*60)
    print("📊 Classes 50-59 (Unseen):")
    _, loader_50_59 = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=1.0, class_range=(50, 59)
    )
    loss_50_59, acc_50_59 = evaluate(model, loader_50_59, criterion, DEVICE)
    print(f"   Accuracy: {acc_50_59 * 100:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print("="*60)
    print(f"   Classes 0-9:    {acc_0_9 * 100:6.2f}%")
    print(f"   Classes 10-49:  {acc_10_49 * 100:6.2f}%")
    print(f"   Classes 50-59:  {acc_50_59 * 100:6.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()