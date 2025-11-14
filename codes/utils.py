# utils.py
import torch
import timm
from codes.model import VisionTransformer
from tqdm import tqdm
from torch import nn

def load_pretrained_timm_weights(model):
    """Load compatible weights from timm pretrained model"""
    print("📥 Loading pretrained weights from timm...")
    timm_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    timm_state = timm_model.state_dict()
    
    model_state = model.state_dict()
    compatible_state = {k: v for k, v in timm_state.items() 
                       if k in model_state and model_state[k].shape == v.shape}
    
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    
    print(f"✅ Loaded {len(compatible_state)}/{len(model_state)} weights from timm")
    print_missing_keys(missing, "Missing")
    print_missing_keys(unexpected, "Unexpected")

def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint file"""
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    
    print(f"✅ Loaded checkpoint")
    print_missing_keys(missing, "Missing")
    print_missing_keys(unexpected, "Unexpected")

def print_missing_keys(keys, label):
    """Print missing or unexpected keys"""
    if keys:
        print(f"⚠️  {label} keys ({len(keys)}):")
        for key in keys:
            print(f"    - {key}")

def print_param_stats(model):
    """Print model parameter statistics"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")

def get_model(num_classes=100, pretrained=True, lora_rank=0, checkpoint_path=None, device='cuda'):
    """
    Create Vision Transformer with optional LoRA and weight loading.
    
    Args:
        num_classes: Output classes
        pretrained: Load timm pretrained weights
        lora_rank: LoRA rank (0=full finetuning)
        checkpoint_path: Custom checkpoint path
        device: Device to load on
    """
    model = VisionTransformer(
        img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, lora_rank=lora_rank
    )
    
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path)
    elif pretrained:
        load_pretrained_timm_weights(model)
    
    model = model.to(device)
    print_param_stats(model)
    
    return model

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader with tqdm."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # wrap dataloader with tqdm
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def add_dustbin_node(model, num_classes=100):
    """Add one extra output node (dustbin) to classification head"""
    old_head = model.head
    new_head = nn.Linear(old_head.in_features, num_classes + 1)
    
    # Copy existing weights
    with torch.no_grad():
        new_head.weight[:num_classes] = old_head.weight
        new_head.bias[:num_classes] = old_head.bias
        # Initialize dustbin node randomly
        nn.init.normal_(new_head.weight[num_classes], std=0.02)
        nn.init.zeros_(new_head.bias[num_classes])
    
    model.head = new_head
    return model

def remove_dustbin_node(model, num_classes=100):
    """Remove dustbin node before saving"""
    old_head = model.head
    new_head = nn.Linear(old_head.in_features, num_classes)
    
    with torch.no_grad():
        new_head.weight.copy_(old_head.weight[:num_classes])
        new_head.bias.copy_(old_head.bias[:num_classes])
    
    model.head = new_head
    return model