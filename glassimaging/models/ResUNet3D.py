# -*- coding: utf-8 -*-

"""
3D with ResNet blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

"""
A Resnet-like UNet body with group normalization. Contains no final classification layer.
"""
class ResUNetBody(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4, num_groups_norm=4):
        super(ResUNetBody, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k
        self.num_groups_norm = num_groups_norm

        self.conv = ConvBlock(inputsize, k, 3, norm_groups=num_groups_norm)
        self.resBlock1 = ResNetBlock(n=2, inputsize=k, normgroups=num_groups_norm)
        self.TD1 = TD(inputsize=k, outputsize=k*2)
        self.resBlock2 = ResNetBlock(n=2, inputsize=k * 2, normgroups=num_groups_norm)
        self.TD2 = TD(inputsize=k*2, outputsize=k*4)
        self.resBlock3 = ResNetBlock(n=2, inputsize=k * 4, normgroups=num_groups_norm)
        self.TD3 = TD(inputsize=k*4, outputsize=k * 4)
        self.resBlock4 = ResNetBlock(n=2, inputsize=k * 4, normgroups=num_groups_norm)
        self.TD4 = TD(inputsize=k*4, outputsize=k*4)
        self.resBlockmid = ResNetBlock(n=2, inputsize=k * 4, normgroups=num_groups_norm)
        self.UP1 = nn.ConvTranspose3d(k*4, k*4, 2, stride=2)
        self.resBlock4_right = ResNetBlock(n=2, inputsize=k * 8, normgroups=num_groups_norm)
        self.UP2 = nn.ConvTranspose3d(k*8, k*4, 2, stride=2)
        self.resBlock3_right = ResNetBlock(n=2, inputsize=k * 8, normgroups=num_groups_norm)
        self.UP3 = nn.ConvTranspose3d(k*8, k*2, 2, stride=2)
        self.resBlock2_right = ResNetBlock(n=2, inputsize=k * 4, normgroups=num_groups_norm)
        self.UP4 = nn.ConvTranspose3d(k*4, k*1, 2, stride=2)
        self.resBlock1_right = ResNetBlock(n=2, inputsize=k * 2, normgroups=num_groups_norm)

    def forward(self, x):
        res = self.conv(x)
        res = self.resBlock1(res)
        skip1 = res.clone()
        res = self.TD1(res)
        res = self.resBlock2(res)
        skip2 = res.clone()
        res = self.TD2(res)
        res = self.resBlock3(res)
        skip3 = res.clone()
        res = self.TD3(res)
        res = self.resBlock4(res)
        skip4 = res.clone()
        res = self.TD4(res)
        res = self.resBlockmid(res)
        res = self.UP1(res)
        skip4 = skip4
        res = torch.cat([res, skip4], dim=1)
        res = self.resBlock4_right(res)
        res = self.UP2(res)
        skip3 = skip3
        res = torch.cat([res, skip3], dim=1)
        res = self.resBlock3_right(res)
        res = self.UP3(res)
        skip2 = skip2
        res = torch.cat([res, skip2], dim=1)
        res = self.resBlock2_right(res)
        res = self.UP4(res)
        skip1 = skip1
        res = torch.cat([res, skip1], dim=1)
        res = self.resBlock1_right(res)
        return res

"""
A full Resnet-like UNet with group normalization. 
"""
class ResUNet(nn.Module):

    def __init__(self, k=16, outputsize=2, inputsize=4, num_groups_norm=4):
        super(ResUNet, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self. k = k
        self.body = ResUNetBody(k=k, outputsize=outputsize, inputsize=inputsize, num_groups_norm=num_groups_norm)
        self.FC = ConvBlock(k*2, k*2, 1, padding=False)
        self.classifier = nn.Conv3d(k*2, self.outputsize, 1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        if k % num_groups_norm != 0:
            warnings.warn("Number of normalization groups {} is not valid for k={}. Set to 1 instead (LayerNorm).".format(num_groups_norm, k))
            num_groups_norm = 1
        self.num_groups_norm = num_groups_norm

    def forward(self, x):
        res = self.body(x)
        res = self.FC(res)
        res = self.classifier(res)
        res = self.softmax(res)
        return res


    @staticmethod
    def initFromDesc(desc):
        inputsize = desc['inputsize']
        outputsize = desc['outputsize']
        k = desc['k'] if 'k' in desc else 16
        num_groups = desc['num_groups'] if 'num_groups' in desc else 4
        return ResUNet(outputsize=outputsize, inputsize=inputsize, k=k, num_groups_norm=num_groups)

    def getDesc(self):
        return ['resunet', {
            'inputsize': self.inputsize,
            'outputsize': self.outputsize,
            'k': self.k,
            'num_groups': self.num_groups_norm
        }]

""" 
Convolution with residual connection. Any number of convolutions with leaky relu and group normalization.
"""
class ResNetBlock(nn.Module):

    def __init__(self, k=10, n=4, inputsize=32, normgroups=4):
        super(ResNetBlock, self).__init__()
        self.k = k
        self.n = n
        self.inputsize = inputsize
        self.convolutions = nn.ModuleList([nn.Conv3d(inputsize, inputsize, 3, padding=1) for _ in range(0, self.n)])
        self.groupNorm = nn.ModuleList([nn.GroupNorm(normgroups, inputsize) for _ in range(0, self.n)])

    def forward(self, x):
        res = x
        for i in range(0, self.n):
            res = self.convolutions[i](res)
            res = self.groupNorm[i](res)
            res = F.leaky_relu(res)
        res.add(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i - (self.n * 2) for i in inputsize]
        return outputsize

""" 
Convolution with leaky relu, group normalization, and optional dropout
"""
class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, dropout=False, norm_groups=4,
                 padding=True):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.groupnorm_layer = nn.GroupNorm(norm_groups, channels_out)
        if padding:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv3d(channels_in, channels_out, kernel_size, padding=padding)
        if dropout:
            self.dropout_layer = nn.Dropout3d(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm_layer(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = F.leaky_relu(x)
        return x


""" 
Transition down module implemented through 3x3x3 convolution with stride=2
"""
class TD(nn.Module):

    def __init__(self, inputsize=32, outputsize=32):
        super(TD, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.convolution = nn.Conv3d(self.inputsize, self.outputsize, 3, stride=2, padding=1)

    def forward(self, x):
        res = self.convolution(x)
        return res

    def getOutputImageSize(self, inputsize):
        outputsize = [i // 2 for i in inputsize]
        return outputsize

    def getOutputChannelSize(self):
        return self.k
