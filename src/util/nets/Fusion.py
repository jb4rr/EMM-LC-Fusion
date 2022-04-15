import torch.nn as nn
import torch
import config
from sys import path
from .DAE import DenoisingAutoEncoder
from .AlignedXception import AlignedXception

path.append('util/nets/')

class Fusion(nn.Module):
    def __init__(self, dae_model=config.DATA_DIR+f'/models/Unimodal/DAE/{config.DAE_NUM}/checkpoints/LAST_DAE.pth'):
        super(Fusion, self).__init__()

        # Images
        BatchNorm = nn.BatchNorm3d
        filters = [16, 32, 64, 128, 128, 256]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Load Pretrained Models
        self.DAE = DenoisingAutoEncoder()
        self.load_DAE_model(dae_model)

        # Combination
        self._fc0 = nn.Linear(filters[-1] * 2 * 2 * 2 + 1480, filters[-1])  # 1480 where N = 20

        self._fc2 = nn.Linear(filters[-1], 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        # ---------For Simple Fusion -----------#
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        # ----------- DAE Feature Fusion ---------- #
        y = self.relu(self.DAE(y)[0])
        y = y.view(y.shape[0], -1)

        # Combination via concatenation
        x = self.relu(self._fc0(torch.cat((x, y), dim=1)))
        x = self.dropout(x)
        x = self._fc2(x)
        return x

    def load_DAE_model(self, dae_model):
        self.DAE.to(device=config.DEVICE)
        self.DAE.float()
        self.DAE.load_state_dict(torch.load(dae_model, map_location=torch.device(config.DEVICE))['state_dict'])
        self.DAE.eval()  # Disable Dropout
        i = 0
        for name, param in self.DAE.named_parameters():
            if i != 3:
                param.requires_grad = False  # Freeze Denoising AutoEncoder


if __name__ == '__main__':
    model = Fusion()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Grad == True:  {name}")
        else:
            print(f"Grad == False: {param.data}")

    img_input, label_input = torch.rand((2, 1, 128, 128, 128)), torch.rand((2, 1, config.NUM_FEATURES))
    out = model(img_input, label_input)
    print(out.shape)
    print(out)