import logging
import math
import torch.nn as nn
from fds import FDS

print = logging.info


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=True):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        return self.layer(x)


class DensityMapRegressor(nn.Module):
    # 初始化 DensityMapRegressor 类
    def __init__(self, in_channels, reduction=16):
        super(DensityMapRegressor, self).__init__()  # 调用基类的初始化方法

        # 根据 reduction 参数的不同，构建不同的回归器结构
        if reduction == 8:
            self.regressor = nn.Sequential(
                # 上采样层，将输入通道数 in_channels 上采样到 128
                UpsamplingLayer(in_channels, 128),
                # 继续上采样到 64
                UpsamplingLayer(128, 64),
                # 再上采样到 32
                UpsamplingLayer(64, 32),
                # 最后通过一个 1x1 卷积层将通道数减少到 1，生成密度图
                nn.Conv2d(32, 1, kernel_size=1),
                # 使用 LeakyReLU 激活函数
                nn.LeakyReLU()
            )
        elif reduction == 16:
            self.regressor = nn.Sequential(
                # 与 reduction == 8 类似，但是最后多一个上采样步骤到 16
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                UpsamplingLayer(32, 16),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.ReLU(inplace=True)
            )

        # 初始化模型参数
        # self.reset_parameters()

    # 前向传播方法，将输入 x 通过回归器处理
    def forward(self, x):
        return self.regressor(x)

    # 参数重置方法，使用特定的初始化方法初始化模型的权重和偏置
    # def reset_parameters(self):
    #     for module in self.modules():  # 遍历模型中所有的模块
    #         if isinstance(module, nn.Conv2d):  # 如果模块是二维卷积层
    #             # 初始化权重为标准正态分布
    #             nn.init.normal_(module.weight, std=0.01)
    #             # 如果存在偏置项，则初始化为常数 0
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)


class ResNet(nn.Module):

    def __init__(self, block, layers, fds, bucket_num, bucket_start, start_update, start_smooth,
                 kernel, ks, sigma, momentum, dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.regressor = DensityMapRegressor(2048, 16)
        self.avgpool1 = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(2809 * block.expansion, 1)
        if fds:
            self.FDS = FDS(
                feature_dim=2809 * block.expansion, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma,
                momentum=momentum
            )
        self.fds = fds
        self.start_smooth = start_smooth

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.regressor(x)
        x = self.avgpool1(x)
        encoding = x.view(x.size(0), -1)

        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        x = self.linear(encoding_s)

        if self.training and self.fds:
            return x, encoding
        else:
            return x


def resnet_regressor(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
