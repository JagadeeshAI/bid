#!/bin/bash

# train_oracle.sh - Train Oracle models on different class ranges
# Usage: bash scripts/train_oracle.sh

echo "🏛️  Starting Oracle Training Pipeline"
echo "======================================"
echo "Training models on different class ranges with full data (ratio=1.0)"
echo "Epochs: 50 | Data Ratio: 1.0"
echo ""

# Create necessary directories
mkdir -p checkpoints/oracle
mkdir -p logs/oracle

# Function to run training with logging
train_model() {
    local range=$1
    local log_file="logs/oracle/${range//-/_}.log"
    
    echo "🚀 Training on classes $range..."
    echo "📝 Logging to: $log_file"
    echo "⏰ Started at: $(date)"
    echo ""
    
    python codes/train.py \
        --range $range \
        --epochs 50 \
        --data-ratio 1.0 \
        --lr 1e-4 \
        --lora-rank 0 \
        | tee $log_file
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed training for classes $range"
    else
        echo "❌ Training failed for classes $range"
        exit 1
    fi
    
    echo "⏰ Finished at: $(date)"
    echo ""
}

# Training schedule - 6 different class ranges
echo "📋 Training Schedule:"
echo "  1. Classes 0-49   (Base range)"
echo "  2. Classes 10-59  (Shifted +10)"
echo "  3. Classes 20-69  (Shifted +20)" 
echo "  4. Classes 30-79  (Shifted +30)"
echo "  5. Classes 40-89  (Shifted +40)"
echo "  6. Classes 50-99  (Final range)"
echo ""

# Start training pipeline
echo "🎯 Starting training pipeline..."
echo "==============================="

# 1. Train on classes 0-49
train_model "0-49"

# 2. Train on classes 10-59  
train_model "10-59"

# 3. Train on classes 20-69
train_model "20-69"

# 4. Train on classes 30-79
train_model "30-79"

# 5. Train on classes 40-89
train_model "40-89"

# 6. Train on classes 50-99
train_model "50-99"

# Summary
echo ""
echo "🎉 Oracle Training Pipeline Completed!"
echo "======================================"
echo "📁 Checkpoints saved in: checkpoints/oracle/"
echo "📝 Logs saved in: logs/oracle/"
echo ""

# List generated files
echo "📊 Generated Checkpoints:"
ls -la checkpoints/oracle/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""

echo "📋 Generated Logs:"
ls -la logs/oracle/*.log 2>/dev/null || echo "  No logs found"
echo ""

echo "⏰ Pipeline completed at: $(date)"
echo "✨ All Oracle models ready for evaluation!"