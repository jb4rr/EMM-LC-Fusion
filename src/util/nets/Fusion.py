import torch.nn as nn
import torch
import config as config
from .DAE import DenoisingAutoEncoder
from .AlignedXception import AlignedXception


class Fusion(nn.Module):
    def __init__(self, dae_model=config.DATA_DIR+'/models/Unimodal/DAE/dae_param_old/N30/checkpoints/Last.pth',
                        alx_model=config.DATA_DIR+'/models/Unimodal/ALX/checkpoints/Best.pth'):
        super(Fusion, self).__init__()

        # Images
        BatchNorm = nn.BatchNorm3d
        filters = [16, 32, 64, 128, 128, 256]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Load Pretrained Models
        self.DAE = DenoisingAutoEncoder()
        self.DAE.load_state_dict(torch.load(dae_model)['state_dict'])
        for dae_param in self.DAE.parameters():
            dae_param.requires_grad = False
        self.DAE.eval()

        #self.fc_d = nn.Linear(74, 512)

        self.ALX = AlignedXception(BatchNorm, filters)
        self.ALX.load_state_dict(torch.load(alx_model)['state_dict'])
        for alx_param in self.ALX.parameters():
            alx_param.requires_grad = False
        self.ALX.eval()

        # Combination
        self._fc0 = nn.Linear(75776 + 2220, filters[-1]) #1480 where N = 20
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(filters[-1], 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        # ----------- ALX Feature Fusion ---------- #
        _, x = self.ALX(x)
        x1, x2, x3 = x

        # ----------- DAE Feature Fusion ---------- #
        y = self.relu(self.DAE(y)[0])
        y = y.view(y.shape[0], -1)
        #y = self.relu(self.fc_d(y))
        #y = torch.squeeze(y, dim=1)

        # Combination via concatenation
        x = self.relu(self._fc0(torch.cat((x1, x2, x3, y), dim=1)))
        x = self._dropout(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    model = Fusion()
    img_input, label_input = torch.rand((2, 1, 128, 128, 128)), torch.rand((2, 1, config.NUM_FEATURES))
    out = model(img_input, label_input)
    print(out.shape)
    print(out)