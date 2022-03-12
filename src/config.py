import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"E:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50
IMAGE_SIZE = np.asarray((64, 64, 64))
N_CLASSES = 1
LR = 0.001

# DAE FACTORS INDEX
# Make sure that Cancer Columns is excluded from this index.
FACT_IDX = [i for i in range(2,79)] #[2,3,4,5,6,7,8,9,10,12,14,15,16]
