#!/usr/bin/env python3
# bid_lora_dustbin.py
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os

from data import get_data_loaders_local
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE
from utils import get_model, evaluate,add_dustbin_node,remove_dustbin_node


def train_dustbin(model, forget_loader, retain_loader, new_loader, forget_val,new_val, retain_val,
                  dustbin_idx, epochs=10, alpha=1.0, beta=1.0, gamma=1.0):
    """Train forget classes to predict dustbin node"""
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🗑️  Training with dustbin node (idx={dustbin_idx})")
    print(f"   α={alpha}, β={beta}, γ={gamma}\n")
    
    for epoch in range(epochs):
        model.train()
        
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        new_iter = iter(new_loader)
        
        max_batches = min(len(forget_loader), len(retain_loader), len(new_loader))
        
        with tqdm(total=max_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in range(max_batches):
                f_data, f_target = next(forget_iter)
                r_data, r_target = next(retain_iter)
                n_data, n_target = next(new_iter)
                
                f_data, f_target = f_data.to(DEVICE), f_target.to(DEVICE)
                r_data, r_target = r_data.to(DEVICE), r_target.to(DEVICE)
                n_data, n_target = n_data.to(DEVICE), n_target.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Forget: predict dustbin
                f_output = model(f_data)
                dustbin_targets = torch.full_like(f_target, dustbin_idx)
                forget_loss = beta * criterion(f_output, dustbin_targets)
                
                # Retain: predict correct class
                r_output = model(r_data)
                retain_loss = alpha * criterion(r_output, r_target)
                
                # New: predict correct class
                n_output = model(n_data)
                new_loss = gamma * criterion(n_output, n_target)
                
                loss = forget_loss + retain_loss + new_loss
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                pbar.set_postfix({
                    'forget': f'{(forget_loss.item()/beta):.3f}',
                    'retain': f'{(retain_loss.item()/alpha):.3f}',
                    'new': f'{(new_loss.item()/gamma):.3f}'
                })
        
        model.eval()
        with torch.no_grad():
                # Forget: check if predicting dustbin
                correct_dustbin, total = 0, 0
                for data, targets in forget_val:
                    data = data.to(DEVICE)
                    outputs = model(data)
                    preds = outputs.argmax(dim=1)
                    correct_dustbin += (preds == dustbin_idx).sum().item()
                    total += targets.size(0)
                dustbin_acc = 100 * correct_dustbin / total
                
                # Retain & New: normal accuracy
                _, acc_r = evaluate(model, retain_val, criterion, DEVICE)
                _, acc_n = evaluate(model, new_val, criterion, DEVICE)
                _, acc_f = evaluate(model, forget_val, criterion, DEVICE)
                
                total_samples = len(retain_loader.dataset) + len(new_loader.dataset)
                acc_overall = (acc_r * len(retain_loader.dataset) + 
                              acc_n * len(new_loader.dataset)) / total_samples
            
        print(f"  Forget→Dustbin:  {dustbin_acc:6.2f}%")
        print(f"  Forget→real:  {acc_f*100:6.2f}%")
        print(f"  Retain (10-49):  {acc_r*100:6.2f}%")
        print(f"  New (50-59):     {acc_n*100:6.2f}%")
        print(f"  Overall (10-59): {acc_overall*100:6.2f}%\n")
    
    return model

def main():
    print("📈 BID-LoRA: Dustbin Unlearning\n")
    
    model = get_model(
        num_classes=100,
        pretrained=False,
        lora_rank=0,
        checkpoint_path="checkpoints/best_0_49.pth",
        device=DEVICE
    )
    
    # Add dustbin node
    model = add_dustbin_node(model, num_classes=100)
    model = model.to(DEVICE)
    dustbin_idx = 100
    print(f"✅ Added dustbin node at index {dustbin_idx}\n")
    
    criterion = nn.CrossEntropyLoss()
    
    # Load data
    forget_train, forget_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=1.0, class_range=(0, 9)
    )
    retain_train, retain_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=0.1, class_range=(10, 49)
    )
    new_train, new_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=1.0, class_range=(50, 59)
    )
    
    # Train
    model = train_dustbin(
        model,
        forget_train,
        retain_train,
        new_train,
        forget_val,
        retain_val,   # FIXED
        new_val,      # FIXED
        dustbin_idx=dustbin_idx,
        epochs=10,
        alpha=0.15,
        beta=0.15,
        gamma=0.70
    )

    
    # Final evaluation
    print("="*60)
    print("📊 FINAL EVALUATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Forget: dustbin accuracy
        correct_dustbin, total = 0, 0
        for data, targets in forget_val:
            data = data.to(DEVICE)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct_dustbin += (preds == dustbin_idx).sum().item()
            total += targets.size(0)
        dustbin_acc = 100 * correct_dustbin / total
        
        _, acc_r = evaluate(model, retain_val, criterion, DEVICE)
        _, acc_n = evaluate(model, new_val, criterion, DEVICE)
        
        total_samples = len(retain_val.dataset) + len(new_val.dataset)
        acc_overall = (acc_r * len(retain_val.dataset) + 
                       acc_n * len(new_val.dataset)) / total_samples
    
    print(f"   Forget→Dustbin:  {dustbin_acc:6.2f}%")
    print(f"   Retain (10-49):  {acc_r*100:6.2f}%")
    print(f"   New (50-59):     {acc_n*100:6.2f}%")
    print(f"   Overall (10-59): {acc_overall*100:6.2f}%")
    print("="*60)
    
    # Remove dustbin and save
    model = remove_dustbin_node(model, num_classes=100)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bid_lora_dustbin_10_59.pth")
    print("\n✅ Removed dustbin, saved: checkpoints/bid_lora_dustbin_10_59.pth")

if __name__ == "__main__":
    main()