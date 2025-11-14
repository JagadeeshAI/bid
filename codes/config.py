import torch

DATA_ROOT = "/media/jag/volD2/cifer100/cifer"
MODEL_NAME = "facebook/deit-tiny-patch16-224"
NUM_CLASSES = 100
EPOCHS = 3
BATCH_SIZE = 64
LR = 5e-4
OUTPUT_DIR = "./checkpoints"
IMG_SIZE = 224
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"