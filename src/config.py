import torch
import numpy as np
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"D:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 2
IMAGE_SIZE = np.asarray((450, 450, 450))
N_CLASSES = 2
