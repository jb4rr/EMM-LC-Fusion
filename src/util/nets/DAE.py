import torch
import torch.nn as nn
from sys import path
import src.config as config

path.append('../../util/')


class DenoisingAutoEncoder(nn.Module):
    '''
    Ref: https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
    '''
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        n = config.NUM_FEATURES
        self.encoder = nn.Sequential(
            # Randomly Dropout 1 Neuron to add 'noise' Default 0.2
            nn.Dropout(0.2),
            nn.Linear(n, n*10),
            nn.Linear(n*10, n*15),
            nn.Linear(n*15, n*20),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n*20, n*15),
            nn.Linear(n*15, n*10),
            nn.Linear(n*10, n),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    clinical_data = torch.rand((1,1,config.NUM_FEATURES), device=config.DEVICE)
    AutoEncoder = DenoisingAutoEncoder().to(config.DEVICE)
    out = AutoEncoder(clinical_data)