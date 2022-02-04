# Title: model.py
# Author: James Barrett
# Date: 31/01/22

from sys import path
import src.config as config
import torch.nn as nn
import torch.nn.functional as F
import torch

# Get access to external python modules
path.append('utils/')


class VGG16(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv2_1 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv3_1 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3_3 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv4_1 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4_3 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv5_1 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5_2 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5_3 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)  # dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # Get Sample data
    image = torch.rand((1, 1, 64, 64, 64), device=config.DEVICE)
    # imin = image.view(64 * 64 * 64, -1)
    # print(imin.shape)
    # 5D: [batch_size, channels, depth, height, width]
    # print(f'INPUT SHAPE: {image.shape} \nINPUT TYPE: {type(image)}')
    vgg_model = VGG16().to(config.DEVICE)
    image = image

    # Test Model with sample data
    output = vgg_model(image)
    print(output)
