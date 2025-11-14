#!/usr/bin/env python3
# bid_lora_dustbin_kd_adaptive.py

import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os

from codes.data import get_data_loaders_local
from codes.config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, DEVICE
from codes.utils import get_model, evaluate, add_dustbin_node, remove_dustbin_node


# =====================================================================
# 1. Adaptive α, β, γ update
# =====================================================================
def update_adaptive_weights(alpha, beta, gamma,
                            acc_r, acc_f, acc_n,
                            n_r=40, n_f=10, n_n=10,           # <--- NEW
                            momentum: float = 0.9,
                            forget_threshold: float = 0.01,
                            retain_min: float = 0.2,
                            eps: float = 1e-8):
    """
    Adaptive asymmetric weights.
    Each component is now scaled by:
        need * (num_classes_in_group)
    """

    # ----------------------------------------------------
    # 1. Class weights (normalize to sum=1)
    # ----------------------------------------------------
    total_classes = n_r + n_f + n_n + eps
    w_r = n_r / total_classes
    w_f = n_f / total_classes
    w_n = n_n / total_classes

    # ----------------------------------------------------
    # 2. Compute base needs
    # ----------------------------------------------------
    retain_need = w_r * max(1.0 - acc_r, eps)
    forget_need = w_f * acc_f
    new_need    = w_n * max(1.0 - acc_n, eps)

    # If forget already perfect → no beta need
    if acc_f < forget_threshold:
        forget_need = 0.0

    # ----------------------------------------------------
    # 3. Normalize to get new α β γ
    # ----------------------------------------------------
    total_need = retain_need + forget_need + new_need + eps

    new_alpha = retain_need / total_need
    new_beta  = forget_need / total_need
    new_gamma = new_need  / total_need

    # ----------------------------------------------------
    # 4. EMA smoothing
    # ----------------------------------------------------
    alpha = momentum * alpha + (1 - momentum) * new_alpha
    beta  = momentum * beta  + (1 - momentum) * new_beta
    gamma = momentum * gamma + (1 - momentum) * new_gamma

    # ----------------------------------------------------
    # 5. Retain should never drop too low
    # ----------------------------------------------------
    if alpha < retain_min:
        deficit = retain_min - alpha
        alpha = retain_min

        scale = 1.0 / (beta + gamma + eps)
        beta *= (1 - deficit * scale)
        gamma *= (1 - deficit * scale)

    # ----------------------------------------------------
    # 6. If forget is complete → transfer β → γ
    # ----------------------------------------------------
    if acc_f < forget_threshold:
        gamma += beta
        beta = 0.0

    # ----------------------------------------------------
    # 7. Final renormalization
    # ----------------------------------------------------
    total = alpha + beta + gamma + eps
    alpha /= total
    beta  /= total
    gamma /= total

    return alpha, beta, gamma

# =====================================================================
# 2. Retention KD (NKD-style)
# =====================================================================
class RetainKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        s_log = torch.log_softmax(student_logits / self.T, dim=1)
        t_prob = torch.softmax(teacher_logits / self.T, dim=1)
        kd = self.kl(s_log, t_prob) * (self.T ** 2)
        ce = self.ce(student_logits, labels)
        return self.alpha * kd + (1 - self.alpha) * ce



# =====================================================================
# 3. Training Loop: Dustbin + KD + Adaptive Weights
# =====================================================================
def train_dustbin_kd(model_student,
                     forget_loader, retain_loader, new_loader,
                     forget_val, new_val, retain_val,
                     dustbin_idx,
                     epochs=10,
                     alpha=0.3, beta=0.3, gamma=0.4):

    # ------------------------------------------------------------
    # Create teacher (frozen snapshot of initial student)
    # ------------------------------------------------------------
    teacher = get_model(
        num_classes=101,
        pretrained=False,
        lora_rank=0,
        checkpoint_path=None,
        device=DEVICE
    )
    teacher.load_state_dict(model_student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    teacher.eval()
    print("📘 Teacher created and frozen.\n")

    # Optimizer & losses
    ce = nn.CrossEntropyLoss()
    kd_retain = RetainKDLoss(temperature=4.0, alpha=0.7)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_student.parameters()),
        lr=5e-4,
        weight_decay=0.01
    )

    print(f"🔧 Initial α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\n")

    # ------------------------------------------------------------
    # Epoch Loop
    # ------------------------------------------------------------
    for epoch in range(epochs):
        model_student.train()

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        new_iter = iter(new_loader)

        max_batches = min(len(forget_loader), len(retain_loader), len(new_loader))

        with tqdm(total=max_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in range(max_batches):

                # Load minibatches
                f_data, f_tgt = next(forget_iter)
                r_data, r_tgt = next(retain_iter)
                n_data, n_tgt = next(new_iter)

                f_data, f_tgt = f_data.to(DEVICE), f_tgt.to(DEVICE)
                r_data, r_tgt = r_data.to(DEVICE), r_tgt.to(DEVICE)
                n_data, n_tgt = n_data.to(DEVICE), n_tgt.to(DEVICE)

                optimizer.zero_grad()

                # ---------------- Forget → Dustbin ----------------
                f_out = model_student(f_data)
                dustbin_labels = torch.full_like(f_tgt, dustbin_idx)
                forget_loss = beta * ce(f_out, dustbin_labels)

                # ---------------- Retain → KD ---------------------
                with torch.no_grad():
                    t_out = teacher(r_data)
                s_out = model_student(r_data)
                retain_loss = alpha * kd_retain(s_out, t_out, r_tgt)

                # ---------------- New → CE ------------------------
                n_out = model_student(n_data)
                new_loss = gamma * ce(n_out, n_tgt)

                # Total loss
                loss = forget_loss + retain_loss + new_loss
                loss.backward()
                optimizer.step()

                # Safe printing (avoid division by zero)
                safe_alpha = alpha if alpha > 1e-8 else 1.0
                safe_beta  = beta  if beta  > 1e-8 else 1.0
                safe_gamma = gamma if gamma > 1e-8 else 1.0

                pbar.update(1)
                pbar.set_postfix({
                    "forget": f"{forget_loss.item()/safe_beta:.3f}",
                    "retain": f"{retain_loss.item()/safe_alpha:.3f}",
                    "new":    f"{new_loss.item()/safe_gamma:.3f}",
                })

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------
        model_student.eval()
        with torch.no_grad():

            # Forget → Dustbin accuracy
            correct_db, total = 0, 0
            for x, y in forget_val:
                x = x.to(DEVICE)
                preds = model_student(x).argmax(1)
                correct_db += (preds == dustbin_idx).sum().item()
                total += y.size(0)
            dustbin_acc = correct_db / total

            # Retain / New / Forget-real
            _, acc_r = evaluate(model_student, retain_val, ce, DEVICE)
            _, acc_n = evaluate(model_student, new_val, ce, DEVICE)
            _, acc_f = evaluate(model_student, forget_val, ce, DEVICE)

            # -------------------- Overall (10–59) --------------------
            full_correct = 0
            full_total = 0

            for x, y in retain_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model_student(x).argmax(1)
                full_correct += (preds == y).sum().item()
                full_total += y.size(0)

            for x, y in new_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model_student(x).argmax(1)
                full_correct += (preds == y).sum().item()
                full_total += y.size(0)

            acc_overall = full_correct / full_total

        # ------------------------------------------------------------
        # Print results
        # ------------------------------------------------------------
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Forget→Dustbin: {100*dustbin_acc:6.2f}%")
        print(f"  Forget→Real:    {acc_f*100:6.2f}%")
        print(f"  Retain (10–49): {acc_r*100:6.2f}%")
        print(f"  New (50–59):    {acc_n*100:6.2f}%")
        print(f"  Overall (10–59):{acc_overall*100:6.2f}%")

        # ------------------------------------------------------------
        # Adaptive α β γ Update
        # ------------------------------------------------------------
        alpha, beta, gamma = update_adaptive_weights(
            alpha, beta, gamma,
            acc_r=acc_r,
            acc_f=acc_f,
            acc_n=acc_n,
            momentum=0.9
        )

        print(f"🧮 Updated α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\n")

    return model_student



# =====================================================================
# 4. Main
# =====================================================================
def main():
    print("📈 BID-LoRA + Teacher–Student KD + Adaptive Weights\n")

    # Load original model
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
    print(f"🗑️ Added dustbin node at index {dustbin_idx}\n")

    # Datasets
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
    model = train_dustbin_kd(
        model,
        forget_train, retain_train, new_train,
        forget_val, new_val, retain_val,
        dustbin_idx=dustbin_idx,
        epochs=20,
        alpha=0.3, beta=0.3, gamma=0.4
    )

    # Save final model
    model = remove_dustbin_node(model, num_classes=100)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bid_lora_dustbin_kd_adaptive.pth")

    print("\n✅ Saved: checkpoints/bid_lora_dustbin_kd_adaptive.pth")


if __name__ == "__main__":
    main()
