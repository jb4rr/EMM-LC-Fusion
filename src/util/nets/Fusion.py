import torch.nn as nn
import torch
import src.config as config

from DAE import DenoisingAutoEncoder
from AlignedXception import AlignedXception


class Fusion(nn.Module):
    def __init__(self, num_classes=2):
        super(Fusion, self).__init__()
        # Images
        BatchNorm = nn.InstanceNorm3d
        filters = [32, 64, 128, 256, 256, 512]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Descriptor
        self.fc_d = DenoisingAutoEncoder()
        # self.fc_d = nn.Linear(76, 512)

        # Combination

        self._fc0 = nn.Linear(filters[-1] * 4 * 4 * 4 + 1480, filters[-1])
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(filters[-1], 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        # Descriptor index [0] for encoded output, index [1] for decoded output
        y = self.relu(self.fc_d(y)[0])
        y = y.view(y.shape[0], -1)

        # Combination
        x = self.relu(self._fc0(torch.cat([x, y], dim=1)))
        x = self._dropout(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    model = Fusion()
    img_input, label_input = torch.rand((1, 1, 256, 256, 256)), torch.rand((1,1,config.NUM_FEATURES))
    out = model(img_input, label_input)
    print(out.shape)
    print(out)