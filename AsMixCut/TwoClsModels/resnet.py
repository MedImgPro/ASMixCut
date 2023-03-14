import os
import math
import torch
import torch.nn as nn
from TwoClsModels.rga_modules import RGA_Module
from torch.nn import functional as F
import torchvision.models
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

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)

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



        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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


    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)



        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)

        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)


        x1 = self.avgpool(x1)

        # print(x.size())  # 这里打印看下输入全连接层前feature map的大小
        x11 = x1.view(x1.size(0), -1)
        # print(x.size())  # 这里打印看下输入全连接层前feature map的大小
        x11 = self.fc(x11)

        return x1


class Three_ResNet(nn.Module):

    def __init__(self, num_classes=2):
        super(Three_ResNet, self).__init__()

        self.feature_extracter1 = resnet18(pretrained=True)
        self.feature_extracter2 = resnet18(pretrained=True)
        self.feature_extracter3 = resnet18(pretrained=True)


        self.fc = nn.Linear(512 * 3, num_classes, bias=False)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        # self.feat_bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)
        # self.cls.apply(weights_init_classifier)


    def forward(self, x1 , x2, x3):
        ####orig Resnet
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x1 = self.feature_extracter1(x1)
        x2 = self.feature_extracter2(x2)
        x3 = self.feature_extracter3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    model_dict = model.state_dict()  # 网络层的参数
    if pretrained:
        # model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
        # model.fc = nn.Linear(model.fc.in_features*3, 2)  # 想输出为2个类别时
        # model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        #
        #print(model)
        pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet18']))
        # pretrained_dict = torch.load('/public/zouqingqing/ImageNet-master/result-dcm/mri_models_1.pth')

        # pretrained_dict = {k.replace('module.', ''): v for k, v in
        #                    pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。

        # 删除pretrained_dict.items()中model所没有的东西
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
        # pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet34']))
        pretrained_dict = torch.load('/public/zouqingqing/ImageNet-master/TwoClsMainResult/CLS_ASP_HEL/contrast/resnet50/F1/mri_models_1_1_3')
        # print(pretrained_dict.items())
        pretrained_dict = {k.replace('module.', ''): v for k, v in
                           pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
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
        # pretrained_dict = torch.load(
        #     '/public/zouqingqing/ImageNet-master/TwoClsMainResult/CLS_ASP_HEL/contrast/resnet50/F5/mri_models_5_20_1')
        # print(pretrained_dict.items())
        # pretrained_dict = {k.replace('module.', ''): v for k, v in
        #                    pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
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
    #resnet = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # model.fc = nn.Linear(model.fc.in_features * 3, 3)  # 想输出为2个类别时
    # model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model_dict = model.state_dict() # 网络层的参数

    model_path='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    if pretrained:
        # resnet.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
        # new_state_dict = resnet.state_dict()
        # dd = model.state_dict()
        # for k in new_state_dict.keys():
        #     print(k)
        #     if k in dd.keys() and not k.startswith('fc'):  # 不加载全连接层
        #         print('yes')
        #         dd[k] = new_state_dict[k]
        # model.load_state_dict(dd)
        # model.fc = nn.Linear(model.fc.in_features*3, 2)  # 想输出为2个类别时
        # model.conv1=nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3,bias=False)


        pretrained_dict = torch.load(os.path.join(models_dir, model_name['resnet101']))
        # pretrained_dict = torch.load('/public/zouqingqing/ImageNet-master/result/YNAS-model_contra/mri_models_4')
        # pretrained_dict = {k.replace('module.', ''): v for k, v in
        #                    pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。

        # 删除pretrained_dict.items()中model所没有的东西
        # pretrained_dict=pretrained_dict['model']
        # print(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值
        model.fc = nn.Linear(model.fc.in_features * 3, 2)  # 想输出为2个类别时
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


def build_model(args):
    return Three_ResNet(args)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class SELayer(nn.Module):
        def __init__(self, channel, reduction=16):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
            y = self.fc(y).view(b, c, 1, 1)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
            return x * y.expand_as(x)  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

# class SpatialAttention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#
#         self.proj_1 = nn.Conv2d(d_model, d_model, 1)
#         self.activation = nn.GELU()
#         self.spatial_gating_unit = AttentionModule(d_model)
#         self.proj_2 = nn.Conv2d(d_model, d_model, 1)
#
#     def forward(self, x):
#         shorcut = x.clone()
#         x = self.proj_1(x)
#         x = self.activation(x)
#         x = self.spatial_gating_unit(x)
#         x = self.proj_2(x)
#         x = x + shorcut
#         return x