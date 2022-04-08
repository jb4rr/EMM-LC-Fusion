import torch
import torch.nn as nn


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1, stride=1,
                 dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride,
                               padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes, affine=True)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
                 BatchNorm=None, start_with_relu=True):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride,
                                  bias=False)
            self.skipbn = BatchNorm(planes, affine=True)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, stride,
                                       BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes, affine=True))
            filters = planes
        else:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, 1, dilation,
                                       BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes, affine=True))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, 1, dilation,
                       BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters, affine=True))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class AlignedXception(nn.Module):
    """
    Modified Aligned Xception
    """
    def __init__(self, BatchNorm, filters):
        super(AlignedXception, self).__init__()
        self.conv1 = nn.Conv3d(1, filters[0], 3, padding=1, stride=2,
                               bias=False)
        self.bn1 = BatchNorm(filters[0], affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(filters[0], filters[0], 3, padding=1,
                               bias=False)
        self.bn2 = BatchNorm(filters[0], affine=True)

        self.block0 = Block(filters[0], filters[1], reps=2, stride=2,
                            BatchNorm=BatchNorm, start_with_relu=False)
        self.block1 = Block(filters[1], filters[2], reps=2, stride=2,
                            BatchNorm=BatchNorm)
        self.block2 = Block(filters[2], filters[3], reps=2, stride=2,
                            BatchNorm=BatchNorm)
        self.block3 = Block(filters[3], filters[4], reps=3, stride=2,
                            BatchNorm=BatchNorm)
        self.block4 = Block(filters[4], filters[5], reps=3, stride=2,
                            BatchNorm=BatchNorm)

        # Final Block for prediction
        self.final = nn.Sequential(nn.Linear(filters[-1] * 2 * 2 * 2, filters[-1]),
                                   nn.Dropout(0.2),
                                   nn.Linear(filters[-1], 2))
        # Init weights
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block0(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # Out1 = Original Output from Daza et al. (2021)
        # Out2 = Prediction output to train feature enrichment
        print("Size")
        rich_features = torch.cat([self.relu(block2.view(block2.size(0), -1)),
                                   self.relu(block3.view(block3.size(0), -1)),
                                   self.relu(block4.view(block4.size(0), -1))], dim=1)
        out1 = self.relu(block4)
        out2 = self.final(out1.view(out1.shape[0], -1))
        return out1, out2, rich_features

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    filters = [16, 32, 64, 128, 128, 256]
    model = AlignedXception(BatchNorm=nn.InstanceNorm3d, filters=filters)
    input = torch.rand(1, 1, 128, 128, 128)
    out1, out2, features = model(input)
    print(features.size())
    print(out1.size())
    print(out2.size())