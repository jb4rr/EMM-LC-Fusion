import torch.nn as nn
import torch
import src.config as config
from .DAE import DenoisingAutoEncoder
from .AlignedXception import AlignedXception



class Fusion(nn.Module):
    def __init__(self, dae_model=config.DATA_DIR+'\\models\\DAE\\checkpoints\\N20\\LAST_DAE.pth'):
        super(Fusion, self).__init__()
        # Images
        BatchNorm = nn.InstanceNorm3d
        filters = [32, 64, 128, 128, 256, 256]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Load Pretrained Model
        DAE = DenoisingAutoEncoder()
        DAE.load_state_dict(torch.load(dae_model)['state_dict'])
        DAE.eval()

        self.fc_d = DAE # == Feature Extraction
        #self.fc_d = nn.Linear(74, 512) # == Simple Feature

        # Combination
        self._fc0 = nn.Linear(filters[-1] * 4 * 4 * 4 + 1480, filters[-1]) #1480 where N = 20
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(filters[-1], 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        # ---------For Simple Fusion -----------#
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        # Descriptor index [0] for encoded output, index [1] for decoded output
        # --------- For Simple Fusion ----------#
        # y = self.relu(self.fc_d(y))
        # y = torch.squeeze(y, dim=1)

        # ----------- For DAE Fusion ---------- #
        y = self.relu(self.fc_d(y)[0])
        y = y.view(y.shape[0], -1)

        # Combination via concatenation
        x = self.relu(self._fc0(torch.cat([x, y], dim=1)))
        x = self._dropout(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    model = Fusion()
    img_input, label_input = torch.rand((2, 1, 128, 128, 128)), torch.rand((2,1,config.NUM_FEATURES))
    out = model(img_input, label_input)
    print(out.shape)
    print(out)