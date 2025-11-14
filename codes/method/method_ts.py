#!/usr/bin/env python3
# bid_lora_dustbin_kd.py

import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os

from codes.data import get_data_loaders_local
from codes.config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE
from codes.utils import get_model, evaluate, add_dustbin_node, remove_dustbin_node


# ---------------------------
# KD Loss for Retention
# ---------------------------
class RetainKDLoss(nn.Module):
    """Teacher-student KD loss for retention (NKD-style)."""
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        s_log_probs = torch.log_softmax(student_logits / self.T, dim=1)
        t_probs = torch.softmax(teacher_logits / self.T, dim=1)
        kd_loss = self.kl(s_log_probs, t_probs) * (self.T ** 2)
        ce_loss = self.ce(student_logits, labels)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


# ---------------------------
# Training with KD for Retention
# ---------------------------
def train_dustbin_kd(model_student, forget_loader, retain_loader, new_loader,
                     forget_val, new_val, retain_val,
                     dustbin_idx, epochs=10, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Modified BID-LoRA Dustbin training with teacher–student retention.
    """

    # Create frozen teacher (copy of initial student)
    teacher = get_model(
        num_classes=101,  # include dustbin node
        pretrained=False,
        lora_rank=0,
        checkpoint_path=None,
        device=DEVICE
    )
    teacher.load_state_dict(model_student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    print("📚 Created teacher (frozen) for retention KD.\n")

    # Optimizer / losses
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_student.parameters()),
        lr=5e-4, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    kd_retain_loss = RetainKDLoss(temperature=4.0, alpha=0.7)

    print(f"🗑️  Training with dustbin node (idx={dustbin_idx})")
    print(f"   α={alpha}, β={beta}, γ={gamma}\n")

    for epoch in range(epochs):
        model_student.train()

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

                # --- Forget loss: predict dustbin ---
                f_out = model_student(f_data)
                dustbin_tgt = torch.full_like(f_target, dustbin_idx)
                forget_loss = beta * criterion(f_out, dustbin_tgt)

                # --- Retain loss: teacher-student KD ---
                with torch.no_grad():
                    t_out = teacher(r_data)
                s_out = model_student(r_data)
                retain_loss = alpha * kd_retain_loss(s_out, t_out, r_target)

                # --- New loss: normal CE ---
                n_out = model_student(n_data)
                new_loss = gamma * criterion(n_out, n_target)

                loss = forget_loss + retain_loss + new_loss
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({
                    'forget': f'{forget_loss.item()/beta:.3f}',
                    'retain': f'{retain_loss.item()/alpha:.3f}',
                    'new': f'{new_loss.item()/gamma:.3f}'
                })

        # ---- Validation per epoch ----
        model_student.eval()
        with torch.no_grad():
            correct_dustbin, total = 0, 0
            for data, targets in forget_val:
                data = data.to(DEVICE)
                outputs = model_student(data)
                preds = outputs.argmax(1)
                correct_dustbin += (preds == dustbin_idx).sum().item()
                total += targets.size(0)
            dustbin_acc = 100 * correct_dustbin / total

            _, acc_r = evaluate(model_student, retain_val, criterion, DEVICE)
            _, acc_n = evaluate(model_student, new_val, criterion, DEVICE)
            _, acc_f = evaluate(model_student, forget_val, criterion, DEVICE)

            total_samples = len(retain_loader.dataset) + len(new_loader.dataset)
            acc_overall = (acc_r * len(retain_loader.dataset) +
                           acc_n * len(new_loader.dataset)) / total_samples

        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Forget→Dustbin:  {dustbin_acc:6.2f}%")
        print(f"  Forget→Real:     {acc_f*100:6.2f}%")
        print(f"  Retain (10-49):  {acc_r*100:6.2f}%")
        print(f"  New (50-59):     {acc_n*100:6.2f}%")
        print(f"  Overall (10-59): {acc_overall*100:6.2f}%\n")

    return model_student


# ---------------------------
# Main
# ---------------------------
def main():
    print("📈 BID-LoRA (Teacher–Student Retain Loss)\n")

    model = get_model(
        num_classes=100,
        pretrained=True,
        lora_rank=0,
        checkpoint_path="checkpoints/best_0_49.pth",
        device=DEVICE
    )

    # Add dustbin node
    model = add_dustbin_node(model, num_classes=100)
    model = model.to(DEVICE)
    dustbin_idx = 100
    print(f"✅ Added dustbin node at index {dustbin_idx}\n")

    # Load data
    forget_train, forget_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, data_ratio=1.0, class_range=(0, 9)
    )
    retain_train, retain_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, data_ratio=0.1, class_range=(10, 49)
    )
    new_train, new_val = get_data_loaders_local(
        DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, data_ratio=1.0, class_range=(50, 59)
    )

    # Train
    model = train_dustbin_kd(
        model, forget_train, retain_train, new_train,
        forget_val, new_val, retain_val,
        dustbin_idx=dustbin_idx,
        epochs=10,
        alpha=0.2,   # retain KD weight
        beta=0.15,   # forget weight
        gamma=0.65   # new weight
    )

    # Final save
    model = remove_dustbin_node(model, num_classes=100)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bid_lora_dustbin_kd_10_59.pth")
    print("\n✅ Saved: checkpoints/bid_lora_dustbin_kd_10_59.pth")


if __name__ == "__main__":
    main()
