import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

#DATA_DIR = r"C:\Users\s1810355\OneDrive - University of Gloucestershire\University\Year 4\Sem 2\CT6039\Data"
DATA_DIR = r"D:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50
IMAGE_SIZE = np.asarray((64, 64, 64))
N_CLASSES = 1
LR = 0.001

# Number of Features
NUM_FEATURES = 74

