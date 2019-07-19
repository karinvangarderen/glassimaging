# -*- coding: utf-8 -*-

from glassimaging.models.ResUNet3D import ResUNet
from glassimaging.models.multipath import UNetMultipath, UNetSharedRep

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def createModel(desc):
    if desc[0] == 'resunet':
        net = ResUNet.initFromDesc(desc[1])
    elif desc[0] == 'multipath':
        net = UNetMultipath.initFromDesc(desc[1])
    elif desc[0] == 'sharedrep':
        net = UNetSharedRep.initFromDesc(desc[1])
    else:
        raise NotImplementedError
    return net
