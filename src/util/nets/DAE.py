import torch
import torch.nn as nn
from sys import path
import config

path.append('util/nets/')


class DenoisingAutoEncoder(nn.Module):
    '''
    Ref: https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
    '''
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        f = config.NUM_FEATURES
        n = [10,20,30]
        self.encoder = nn.Sequential(
            # Dropout Layer == Input Noise
            nn.Dropout(0.2),
            nn.Linear(f, f*n[0]),
            nn.Linear(f*n[0], f*n[1]),
            nn.Linear(f*n[1], f*n[2]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(f*n[2], f*n[1]),
            nn.Linear(f*n[1], f*n[0]),
            nn.Linear(f*n[0], f),
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