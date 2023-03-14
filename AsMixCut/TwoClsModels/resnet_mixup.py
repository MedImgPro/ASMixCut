import os
from torch import einsum
import torch.nn as nn
from mixup import to_one_hot, mixup_process
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import random
import math
from TwoClsModels.GR import *
from einops import rearrange

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)

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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x1,x2,x3, target=None, mixup=False,mixup_hidden=False,args=None,grad1=None,grad2=None,grad3=None,noise1=None, noise2=None,noise3=None,adv_mask1=0,adv_mask2=0,mp=None):
        if mixup_hidden:
            layer_mix = random.randint(0, 4)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None
        if target is not None:
            target_reweighted = to_one_hot(target, 2)

        if layer_mix == 0:
            x1, x2, x3, target_reweighted = mixup_process(x1,
                                                          x2,
                                                          x3,
                                                          target_reweighted,
                                                          args=args,
                                                          grad=grad1,
                                                          noise=noise1,
                                                          adv_mask1=adv_mask1,
                                                          adv_mask2=adv_mask2,
                                                          mp=mp)

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)

        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)
        x3 = self.maxpool(x3)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        x3 = self.layer1(x3)


        if layer_mix == 1:
            x1, x2, x3, target_reweighted = mixup_process(x1, x2, x3, target_reweighted, args=args, hidden=True)


        x1 = self.layer2(x1)
        x2 = self.layer2(x2)
        x3 = self.layer2(x3)


        if layer_mix == 2:
            x1, x2, x3, target_reweighted = mixup_process(x1, x2, x3, target_reweighted, args=args, hidden=True)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)
        x3 = self.layer3(x3)

        if layer_mix == 3:
            x1, x2, x3, target_reweighted = mixup_process(x1, x2, x3, target_reweighted, args=args, hidden=True)


        x1 = self.layer4(x1)
        x2 = self.layer4(x2)
        x3 = self.layer4(x3)

        if layer_mix == 4:
            x1, x2, x3, target_reweighted = mixup_process(x1, x2, x3, target_reweighted, args=args, hidden=True)

        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x3 = self.avgpool(x3)

####融合之后
        x = torch.cat([x1, x2, x3], dim=1)   ###仅融合三序列特征
        # print(x.size())  # 这里打印看下输入全连接层前feature map的大小
        x = x.view(x.size(0), -1)
        # print(x.size())  # 这里打印看下输入全连接层前feature map的大小
        x = self.fc(x)

        if target is not None:
            return x, target_reweighted
        else:
            return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    model_dict = model.state_dict()  # 网络层的参数
    if pretrained:

        pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet18']))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    encoder = nn.Sequential(*list(model.children())[:7])
    model_dict = encoder.state_dict()  # 网络层的参数
    if pretrained:
        pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet34']))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        model.load_state_dict(model_dict,False)  # model加载dict中的数据，更新网络的初始值


    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()  # 网络层的参数
    if pretrained:
        pretrained_dict =torch.load(os.path.join(models_dir, model_name['resnet50']))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值



    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model_dict = model.state_dict() # 网络层的参数
    if pretrained:
        pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet101']))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model

