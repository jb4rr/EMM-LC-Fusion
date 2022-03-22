import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"E:\University of Gloucestershire\Year 4\Dissertation\Data"
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 50
IMAGE_SIZE = np.asarray((64, 64, 64))
N_CLASSES = 1
LR = 0.001

# Number of Features
NUM_FEATURES = 74


