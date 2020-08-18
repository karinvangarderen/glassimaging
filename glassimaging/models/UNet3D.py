# -*- coding: utf-8 -*-

"""
3D with ResNet blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBody(nn.Module):

    def __init__(self, k=32, outputsize=2, inputsize=1):
        super(UNetBody, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k

        self.conv = ConvBlock(inputsize, k, 3)
        self.convBlock1 = ConvBlock(k, k*2, 3)
        self.TD1 = nn.MaxPool3d(2)
        self.convBlock2 = ConvBlock(k*2, k*2, 3)
        self.convBlock21 = ConvBlock(k*2, k*4, 3)
        self.TD2 = nn.MaxPool3d(2)
        self.convBlock3 = ConvBlock(k*4, k*4, 3)
        self.convBlock31 = ConvBlock(k*4, k*8, 3)
        self.TD3 = nn.MaxPool3d(2)
        self.convBlock4 = ConvBlock(k*8,k*8,3)
        self.convBlock41 = ConvBlock(k*8,k*16,3)
        self.UP43 = nn.ConvTranspose3d(k*16, k*16, 2, stride=2)
        self.convBlock3_right = ConvBlock(k*24, k*8, 3)
        self.convBlock31_right = ConvBlock(k*8, k*8, 3)
        self.UP32 = nn.ConvTranspose3d(k*8, k*8, 2, stride=2)
        self.convBlock2_right = ConvBlock(k*12,k*4, 3)
        self.convBlock21_right = ConvBlock(k*4,k*4, 3)
        self.UP21 = nn.ConvTranspose3d(k*4, k*4, 2, stride=2)
        self.convBlock1_right = ConvBlock(k*6, k*2,3)
        self.convBlock11_right = ConvBlock(k*2, k*2,3)

    def forward(self, x):
        res = self.conv(x)
        res = self.convBlock1(res)
        skip1 = res.clone()
        res = self.TD1(res)
        res = self.convBlock2(res)
        res = self.convBlock21(res)
        skip2 = res.clone()
        res = self.TD2(res)
        res = self.convBlock3(res)
        res = self.convBlock31(res)
        skip3 = res.clone()
        res = self.TD3(res)
        res = self.convBlock4(res)
        res = self.convBlock41(res)
        res = self.UP43(res)
        res = torch.cat([res, skip3], dim=1)
        res = self.convBlock3_right(res)
        res = self.convBlock31_right(res)
        res = self.UP32(res)
        res = torch.cat([res, skip2], dim=1)
        res = self.convBlock2_right(res)
        res = self.convBlock21_right(res)
        res = self.UP21(res)
        res = torch.cat([res, skip1], dim=1)
        res = self.convBlock1_right(res)
        res = self.convBlock11_right(res)
        return res




class UNet3D(nn.Module):

    def __init__(self, k=32, outputsize=2, inputsize=1):
        super(UNet3D, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k
        self.body = UNetBody(k=k, outputsize=outputsize, inputsize=inputsize)
        self.classifier = nn.Conv3d(k*2, self.outputsize, 1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.body(x)
        res = self.classifier(res)
        res = self.softmax(res)
        return res


    @staticmethod
    def initFromDesc(desc):
        inputsize = desc['inputsize']
        outputsize = desc['outputsize']
        k = desc['k'] if 'k' in desc else 32
        return UNet3D(outputsize=outputsize, inputsize=inputsize, k=k)

    def getDesc(self):
        return ['3dunet', {
            'inputsize': self.inputsize,
            'outputsize': self.outputsize,
            'k': self.k
        }]

class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, batchnorm=True, padding=True):
        super(ConvBlock, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm3d(channels_out)
        if padding:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv3d(channels_in, channels_out, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        x = F.relu(x)
        return x

