import torch.nn as nn


def conv_3x3_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


def dws_conv_3x3_bn(in_channels, out_channels, dw_stride):
    """
    Depthwise Separable Convolution
    :param in_channels: depthwise conv input channels
    :param out_channels: Separable conv output channels
    :param dw_stride: depthwise conv stride
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=in_channels,
                  kernel_size=3,
                  stride=dw_stride,
                  padding=1,
                  groups=in_channels,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1,
                  stride=1,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    """
    mobilenet V1 implementation. Modify a bit for CIFAR10
    """
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)]  # change stride 2->1 for cifar10
        dws_conv_config = [
            # num, in_channels, out_channels, stride
            [1, 32, 64, 1],
            [1, 64, 128, 1],  # change stride 2->1 for cifar10
            [1, 128, 128, 1],
            [1, 128, 256, 2],
            [1, 256, 256, 1],
            [1, 256, 512, 2],
            [5, 512, 512, 1],
            [1, 512, 1024, 2],
            [1, 1024, 1024, 1]
        ]
        for num, in_channels, out_channels, dw_stride in dws_conv_config:
            for i in range(num):
                layers.append(dws_conv_3x3_bn(in_channels, out_channels, dw_stride))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        y = self.layers(x)
        # print(y.shape)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()
        assert stride == 1 or stride == 2
        self.stride = stride
        self.residual = self.stride == 1 and (in_channels == out_channels)
        expansion_channels = in_channels * expansion_factor

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=expansion_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=expansion_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=expansion_channels,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            return y + x
        else:
            return y


class MobileNetV2(nn.Module):
    """
    mobilenet V2 implementation. modify a bit for CIFAR10
    """
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)] # change stride 2->1 for cifar10
        in_channels = 32
        inverted_residual_block_config = [
            # expansion factor, out_channels, stride
            [1, 16, 1],

            [6, 24, 1],  # change stride 2->1 for cifar10
            [6, 24, 1],

            [6, 32, 2],
            [6, 32, 1],
            [6, 32, 1],

            [6, 64, 2],
            [6, 64, 1],
            [6, 64, 1],
            [6, 64, 1],

            [6, 96, 1],
            [6, 96, 1],
            [6, 96, 1],

            [6, 160, 2],
            [6, 160, 1],
            [6, 160, 1],

            [6, 320, 1],
        ]
        for expansion_factor, out_channels, stride in inverted_residual_block_config:
            layers.append(InvertedResidualBlock(in_channels, out_channels, stride, expansion_factor))
            in_channels = out_channels
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=320,
                      out_channels=1280,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=1280),
            nn.ReLU6(inplace=True),
        ))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        y = self.layers(x)
        # print(y.shape)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y
