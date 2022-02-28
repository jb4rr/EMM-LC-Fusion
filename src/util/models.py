# Title: models.py
# Author: James Barrett
# Date: 31/01/22

from sys import path
import src.config as config
import torch.nn as nn
import torch.nn.functional as F
import torch

# Get access to external python modules
path.append('util/')


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
        x = F.dropout3d(x, 0.5)
        x = F.relu(self.fc2(x))
        #x = F.dropout3d(x, 0.5)
        x = self.fc3(x)
        return x


class DenoisingAutoEncoder(nn.Module):
    '''
    Ref: https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
    '''
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(76, 76),
            nn.ReLU(True),
            nn.Linear(76, 76),
            # Randomly Dropout 1 Neuron to add 'noise'
            nn.Dropout(1 / 76),
            nn.ReLU(True),
            nn.Linear(76, 76*20),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(76*20, 76),
            nn.ReLU(True),
            nn.Linear(76, 76),
            nn.ReLU(True),
            nn.Linear(76, 76),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


if __name__ == '__main__':
    # Test Model
    ct_scan = torch.rand((1, 1, 64, 64, 64), device=config.DEVICE)
    clinical_data = torch.rand((1,1,76), device=config.DEVICE)
    vgg_model = VGG16().to(config.DEVICE)
    AutoEncoder = DenoisingAutoEncoder().to(config.DEVICE)
    # Test Model with sample data
    output_1 = vgg_model(ct_scan)
    output_2 = AutoEncoder(clinical_data)
