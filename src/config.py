import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"/content/gdrive/MyDrive/Colab Notebooks/Project/Data"
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 500
IMAGE_SIZE = np.asarray((64, 64, 64))
N_CLASSES = 1
LR = 0.001

# Number of Features
NUM_FEATURES = 74


DAE_NUM = "N20"