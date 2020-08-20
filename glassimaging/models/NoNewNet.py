# -*- coding: utf-8 -*-

"""
3D with ResNet blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBody(nn.Module):

    def __init__(self, k=30, outputsize=2, inputsize=1):
        super(UNetBody, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k

        self.conv = ConvBlock(inputsize, k, 3)
        self.convBlock1 = ConvBlock(k, k, 3)
        self.TD1 = nn.MaxPool3d(2)
        self.convBlock2 = ConvBlock(k, k*2, 3)
        self.convBlock21 = ConvBlock(k*2, k*2, 3)
        self.TD2 = nn.MaxPool3d(2)
        self.convBlock3 = ConvBlock(k*2, k*4, 3)
        self.convBlock31 = ConvBlock(k*4, k*4, 3)
        self.TD3 = nn.MaxPool3d(2)
        self.convBlock4 = ConvBlock(k*4,k*8,3)
        self.convBlock41 = ConvBlock(k*8,k*8,3)
        self.TD4 = nn.MaxPool3d(2)
        self.convBlock5 = ConvBlock(k*8,k*16,3)
        self.convBlock51 = ConvBlock(k*16,k*8,3)
        
        self.UP54 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock4_right = ConvBlock(k*16, k*8, 3)
        self.convBlock41_right = ConvBlock(k*8, k*4, 3)
        self.UP43 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock3_right = ConvBlock(k*8, k*4, 3)
        self.convBlock31_right = ConvBlock(k*4, k*2, 3)
        self.UP32 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock2_right = ConvBlock(k*4,k*2, 3)
        self.convBlock21_right = ConvBlock(k*2,k*1, 3)
        self.UP21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock1_right = ConvBlock(k*2, k*1,3)
        self.convBlock11_right = ConvBlock(k*1, k*1,3)

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
        skip4 = res.clone()

        res = self.TD4(res)
        res = self.convBlock5(res)
        res = self.convBlock51(res)
        res = self.UP54(res)
        res = torch.cat([res, skip4], dim=1)
        res = self.convBlock4_right(res)
        res = self.convBlock41_right(res)
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




class NoNewNet(nn.Module):

    def __init__(self, k=30, outputsize=2, inputsize=1):
        super(NoNewNet, self).__init__()
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.k = k
        self.body = UNetBody(k=k, outputsize=outputsize, inputsize=inputsize)
        self.classifier = nn.Conv3d(k, self.outputsize, 1, padding=0)
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
        k = desc['k'] if 'k' in desc else 30
        return NoNewNet(outputsize=outputsize, inputsize=inputsize, k=k)

    def getDesc(self):
        return ['nonewnet', {
            'inputsize': self.inputsize,
            'outputsize': self.outputsize,
            'k': self.k
        }]

class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, batchnorm=True, padding=True):
        super(ConvBlock, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layer = nn.InstanceNorm3d(channels_out)
        if padding:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv3d(channels_in, channels_out, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        x = F.leaky_relu(x)
        return x

