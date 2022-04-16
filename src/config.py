import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"/content/gdrive/MyDrive/Colab Notebooks/Project/Data"
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 50
LR = 0.001
NUM_FEATURES = 74


