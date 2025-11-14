# BID-LoRA: Bidirectional LoRA for Machine Unlearning with Vision Transformers

A PyTorch implementation of machine unlearning techniques using Vision Transformers with LoRA (Low-Rank Adaptation) for CIFAR-100 classification. This project explores various unlearning methodologies including dustbin classification, knowledge distillation, and adaptive weight optimization.

## 🎯 Project Overview

This repository implements advanced machine unlearning techniques to selectively "forget" specific classes from a pre-trained Vision Transformer while retaining performance on other classes and learning new ones. The project is structured around three main scenarios:

- **Forget Classes**: 0-9 (classes to be unlearned)
- **Retain Classes**: 10-49 (classes to maintain performance on)
- **New Classes**: 50-59 (new classes to learn)

## 🏗️ Architecture

### Core Components

- **Vision Transformer**: Custom ViT implementation with LoRA support
- **LoRA Integration**: Efficient fine-tuning using Low-Rank Adaptation
- **Dustbin Classification**: Novel approach for class forgetting
- **Knowledge Distillation**: Teacher-student framework for retention
- **Adaptive Weighting**: Dynamic loss balancing during training

## 📁 Repository Structure

```
.
├── codes/                          # Main source code
│   ├── config.py                   # Configuration settings
│   ├── data.py                     # Data loading and preprocessing
│   ├── model.py                    # Vision Transformer implementation
│   ├── train.py                    # Standard training pipeline
│   ├── eval.py                     # Model evaluation utilities
│   ├── utils.py                    # Helper functions and model utilities
│   └── method/                     # Unlearning methodologies
│       ├── method.py               # Basic dustbin unlearning
│       ├── method_ts.py            # Teacher-student knowledge distillation
│       └── adaptive_method.py      # Adaptive weight optimization
├── abalistion/                     # Ablation studies
│   └── ts/                         # Teacher-student experiments
│       ├── ts.py                   # Teacher-student training
│       └── plot.py                 # Results visualization
├── checkpoints/                    # Model checkpoints
└── logs/                          # Training logs
```

## 🚀 Key Features

### 1. Vision Transformer with LoRA
- Custom ViT implementation optimized for CIFAR-100
- Integrated LoRA layers for efficient parameter updates
- Flexible rank configuration for different adaptation strategies

### 2. Dustbin Unlearning Methodology
- Novel "dustbin" node approach for class forgetting
- Forces forgotten classes to predict a special dustbin category
- Maintains model architecture while achieving selective forgetting

### 3. Knowledge Distillation Framework
- Teacher-student setup preserving original knowledge
- NKD (Normalized Knowledge Distillation) implementation
- Feature-level and logit-level distillation support

### 4. Adaptive Weight Optimization
- Dynamic loss weight adjustment during training
- Balances forgetting, retention, and new learning objectives
- Momentum-based smoothing for stable convergence

## ⚙️ Configuration

Key configuration parameters in `codes/config.py`:

```python
DATA_ROOT = "/media/jag/volD2/cifer100/cifer"  # Dataset path
MODEL_NAME = "facebook/deit-tiny-patch16-224"  # Base model
NUM_CLASSES = 100                              # CIFAR-100 classes
EPOCHS = 3                                     # Training epochs
BATCH_SIZE = 64                               # Batch size
LR = 5e-4                                     # Learning rate
IMG_SIZE = 224                                # Input image size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🔧 Usage

### 1. Standard Training (Classes 0-49)

```bash
python codes/train.py
```

Trains the base model on CIFAR-100 classes 0-49, saving the best checkpoint.

### 2. Basic Dustbin Unlearning

```bash
python codes/method/method.py
```

Implements basic dustbin unlearning to forget classes 0-9 while retaining 10-49 and learning 50-59.

### 3. Teacher-Student Knowledge Distillation

```bash
python codes/method/method_ts.py
```

Uses knowledge distillation to preserve performance on retained classes during unlearning.

### 4. Adaptive Weight Optimization

```bash
python codes/method/adaptive_method.py
```

Advanced method with adaptive loss weighting for optimal balance between objectives.

### 5. Model Evaluation

```bash
python codes/eval.py
```

Evaluates trained models across different class ranges to measure unlearning effectiveness.

### 6. Teacher-Student Ablation Study

```bash
python abalistion/ts/ts.py
```

Conducts ablation studies comparing different distillation strategies.

## 📊 Evaluation Metrics

The project evaluates models using several key metrics:

- **Forget Accuracy**: How well the model "forgets" classes 0-9
- **Retain Accuracy**: Performance maintenance on classes 10-49  
- **New Learning Accuracy**: Performance on newly learned classes 50-59
- **Overall Accuracy**: Combined performance on classes 10-59
- **Dustbin Accuracy**: Percentage of forgotten classes classified as dustbin

## 🧪 Experimental Results

The implementation includes comprehensive experimental validation:

### Unlearning Effectiveness
- Successful forgetting of target classes (0-9)
- Maintained performance on retained classes (10-49)
- Effective learning of new classes (50-59)

### Method Comparison
- Basic dustbin vs. knowledge distillation approaches
- Adaptive weighting vs. fixed loss balancing
- Teacher-student frameworks vs. direct optimization

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- timm (PyTorch Image Models)
- loralib
- tqdm
- matplotlib (for visualization)

## 🔍 Implementation Details

### LoRA Configuration
- Configurable rank for adaptation efficiency
- Applied to feed-forward layers in transformer blocks
- Automatic freezing/unfreezing based on rank settings

### Dustbin Methodology
- Additional output node for forgotten classes
- Cross-entropy loss optimization for dustbin prediction
- Automatic dustbin removal after training completion

### Adaptive Weighting Algorithm
- Class-aware weight computation
- Momentum-based smoothing for stability
- Dynamic threshold adjustment for forgetting completion

## 📈 Future Work

- Extension to other architectures (ResNet, ConvNext)
- Multi-stage unlearning for complex scenarios  
- Integration with federated learning frameworks
- Theoretical analysis of unlearning guarantees

## 📄 License

This project is available for research purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This implementation is designed for research purposes and demonstrates various machine unlearning techniques with Vision Transformers. The methods can be adapted for different datasets and architectures based on specific requirements.