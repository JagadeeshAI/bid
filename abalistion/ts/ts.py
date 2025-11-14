#!/usr/bin/env python3
"""
Train student model (10–49) using ViTKD + NKD distillation method.
Student initialized from best_0_49.pth (previously trained 0–49 model).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_data_loaders_local
from utils import get_model, evaluate
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE


# ------------------------------
# 1. Distillation Losses (ViTKD + NKD simplified)
# ------------------------------

class ViTKDLoss(nn.Module):
    """Feature-based distillation (ViTKD style: mimic shallow + generate deep)."""
    def __init__(self, feat_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.feat_weight = feat_weight

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        # Mimic shallow features directly, generate deep (approximate)
        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            if i < len(student_feats) // 2:  # shallow
                loss += self.mse(s_feat, t_feat.detach())
            else:  # deep layers – generation like mapping
                loss += self.mse(torch.tanh(s_feat), torch.tanh(t_feat.detach()))
        return self.feat_weight * loss


class NKDLoss(nn.Module):
    """Logit-based normalized KD (NKD)."""
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Normalize logits as per NKD paper
        s_log_probs = torch.log_softmax(student_logits / self.T, dim=1)
        t_probs = torch.softmax(teacher_logits / self.T, dim=1)
        kd_loss = self.kl(s_log_probs, t_probs) * (self.T ** 2)

        ce_loss = self.ce(student_logits, labels)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


# ------------------------------
# 2. Training Loop
# ------------------------------

def train_student(model_s, model_t, train_loader, val_loader, epochs=30, lr=1e-4, feat_layers=4):
    optimizer = optim.AdamW(model_s.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    kd_feat_loss = ViTKDLoss(feat_weight=1.0)
    kd_logit_loss = NKDLoss(temperature=4.0, alpha=0.7)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model_s.train()
        model_t.eval()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward (teacher/student)
            with torch.no_grad():
                teacher_out = model_t(images)

            student_out = model_s(images)

            # Feature extraction (assuming model exposes intermediate features)
            # If not, replace with .forward_features() or hooks in get_model()
            student_feats = getattr(model_s, 'features', [student_out])
            teacher_feats = getattr(model_t, 'features', [teacher_out])

            loss_feat = kd_feat_loss(student_feats[:feat_layers], teacher_feats[:feat_layers])
            loss_logit = kd_logit_loss(student_out, teacher_out, labels)
            # loss = loss_feat + loss_logit
            loss =  loss_logit

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = student_out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss/(pbar.n+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        # Validation each epoch
        val_loss, val_acc = evaluate(model_s, val_loader, nn.CrossEntropyLoss(), DEVICE)
        scheduler.step()

        print(f"\nEpoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {100.*correct/total:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_s.state_dict(), "checkpoints/best_10_49_student.pth")
            print(f"✅ Saved new best model (Val Acc: {val_acc:.2f}%)")

    print(f"\n🎯 Training complete — Best Val Acc: {best_val_acc:.2f}%")


# ------------------------------
# 3. Main
# ------------------------------

def main():
    class_range = (10, 49)

    print(f"\n🚀 Training Student for Classes {class_range} using ViTKD+NKD")
    print(f"📍 Device: {DEVICE}")

    # Data
    train_loader, val_loader = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
        data_ratio=0.1, class_range=class_range
    )

    # Teacher (frozen) — pre-trained 0–49 model
    teacher = get_model(
        num_classes=100,
        pretrained=False,
        lora_rank=0,
        checkpoint_path="checkpoints/best_0_49.pth",
        device=DEVICE
    )
    teacher.eval()

    # Student (learnable)
    student = get_model(
        num_classes=100,
        pretrained=True,
        lora_rank=0,
        checkpoint_path=None,
        device=DEVICE
    )

    train_student(student, teacher, train_loader, val_loader, epochs=30, lr=1e-4)


if __name__ == "__main__":
    main()
