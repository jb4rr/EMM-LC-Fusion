import torch.nn as nn
import torch
import src.config as config
from .DAE import DenoisingAutoEncoder
from .AlignedXception import AlignedXception


class Fusion(nn.Module):
    def __init__(self, dae_model=config.DATA_DIR+'\\models\\Unimodal\\DAE\\checkpoints\\N20\\BEST_DAE.pth',
                        alx_model=config.DATA_DIR+'\\models\\Unimodal\\ALX\\checkpoints\\BEST 3F.pth'):
        super(Fusion, self).__init__()

        # Images
        BatchNorm = nn.InstanceNorm3d
        filters = [16, 32, 64, 128, 128, 256]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Load Pretrained Models
        DAE = DenoisingAutoEncoder()
        DAE.load_state_dict(torch.load(dae_model)['state_dict'])
        DAE.eval()

        ALX = AlignedXception(BatchNorm, filters)
        ALX.load_state_dict(torch.load(alx_model)['state_dict'])
        ALX.eval()

        self.DAE = DAE  # == Feature Extraction
        self.ALX = ALX
        #self.fc_d = nn.Linear(74, 512) # == Simple Feature

        # Combination
        self._fc0 = nn.Linear(10240 + 1480, filters[-1]) #1480 where N = 20
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(filters[-1], 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        # ---------For Simple Fusion -----------#
        # x = self.backbone(x)
        # x = x.view(x.shape[0], -1)

        # Descriptor index [0] for encoded output, index [1] for decoded output
        # --------- For Simple Fusion ----------#
        # y = self.relu(self.fc_d(y))
        # y = torch.squeeze(y, dim=1)

        # ----------- ALX Feature Fusion ---------- #
        _, _, x = self.ALX(x)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # ----------- DAE Feature Fusion ---------- #
        y = self.relu(self.DAE(y)[0])
        y = y.view(y.shape[0], -1)


        # Combination via concatenation
        x = self.relu(self._fc0(torch.cat((x2, x3, y), dim=1)))
        x = self._dropout(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    model = Fusion()
    img_input, label_input = torch.rand((2, 1, 128, 128, 128)), torch.rand((2, 1, config.NUM_FEATURES))
    out = model(img_input, label_input)
    print(out.shape)
    print(out)