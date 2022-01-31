# Title: model.py
# Author: James Barrett
# Date: 31/01/22

from sys import path
from src.utils.dataloader import LUCASDataset
import torch.nn as nn
import torch

# Get access to external python modules
path.append('utils/')


class VGG16(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        return out


if __name__ == '__main__':
    # Get Sample data
    dataset = LUCASDataset('train_file.csv')
    image = torch.tensor(dataset[0]['image']).unsqueeze(1)
    print(torch.cuda.is_available)
    print(f'INPUT SHAPE: {image.shape} \nINPUT TYPE: {type(image)}')

    # Test Model with sample data
    _, outputs = VGG16().cuda()(image)
    print(outputs)
