import torch
import numpy as np
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"D:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 2000
IMAGE_SIZE = np.asarray((256, 256, 256))
N_CLASSES = 1
LR = 0.001
