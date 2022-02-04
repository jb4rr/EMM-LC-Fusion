import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = r"D:\University of Gloucestershire\Year 4\Dissertation"
BATCH_SIZE = 4
IMAGE_SIZE = [256, 256, 256]
N_CLASSES = 2
