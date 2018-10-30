import torch.nn as nn

_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)


class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name):
        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, 10)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        # y = self.fc2(y)
        # y = self.fc3(y)
        return y


def VGG11():
    return _VGG('VGG11')


def VGG13():
    return _VGG('VGG13')


def VGG16():
    return _VGG('VGG16')


def VGG19():
    return _VGG('VGG19')
