# Title: model.py
# Author: James Barrett
# Date: 31/01/22

from sys import path
import torch.nn as nn
import torch

# Get access to external python modules
path.append('utils/')

class VGG16(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG16, self).__init__()

        # -------Define 3D Network------ #

    def forward(self, x):
        return x


if __name__ == '__main__':
    # Get Sample data
    image = torch.rand((1, 1, 128, 128, 128))
    print(f'INPUT SHAPE: {image.shape} \nINPUT TYPE: {type(image)}')
    vgg_model = VGG16()
    if torch.cuda.is_available():
        vgg_model = vgg_model.cuda()
        image = image.cuda()

    # Test Model with sample data
    output = vgg_model(image)
    print(output)
