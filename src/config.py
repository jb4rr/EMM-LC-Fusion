import torch
import numpy as np
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"E:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 2
NUM_WORKERS = 8
NUM_EPOCHS = 200
IMAGE_SIZE = np.asarray((256, 256, 256))
N_CLASSES = 1
LR = 0.001
