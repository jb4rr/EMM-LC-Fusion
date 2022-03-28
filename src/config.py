import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_DIR = "E:\\University of Gloucestershire\\Year 4\\Dissertation"
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 50
LR = 0.001
NUM_FEATURES=76


