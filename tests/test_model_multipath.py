import torch
import unittest
import numpy as np
from glassimaging.models.multipath import UNetMultipath, UNetSharedRep
from glassimaging.models.ResUNet3D import ResUNet
from glassimaging.models.diceloss import DiceLoss


class TestTorchnet(unittest.TestCase):

    def testRunModel(self):
        net = UNetMultipath(inputsize=4, outputsize=2, k=4)
        net.eval()
        inputshape = [64,64,64]
        inputimage = torch.randn(1,4,*inputshape)
        output = net(inputimage).detach().numpy()
        outputshape = list(output.shape[2:5])

    def testLoadModels(self):
        basemodel = ResUNet(inputsize=1, outputsize=2, k=4)
        state_dict = basemodel.state_dict()
        net = UNetMultipath(inputsize=4, outputsize=2, k=4)
        list_of_dicts = [state_dict] * 4
        net.loadExistingModels(list_of_dicts)
        inputshape = [64,64,64]
        inputimage = torch.randn(1,4,*inputshape)
        output = net(inputimage).detach().numpy()

    def testUNet(self):
        basemodel = ResUNet(inputsize=1, outputsize=2, k=4)
        inputshape = [64,64,64]
        inputimage = torch.randn(1,1,*inputshape)
        output = basemodel(inputimage).detach().numpy()

    def testRunModel(self):
        net = UNetSharedRep(inputsize=4, outputsize=2, k=4)
        net.eval()
        inputshape = [64,64,64]
        inputimage = torch.randn(1,4,*inputshape)
        output = net(inputimage).detach().numpy()
        outputshape = list(output.shape[2:5])

    def testDiceLoss(self):
        inputimage = torch.zeros(1, 5, 20, 20, 20)
        outputimage = torch.zeros(1, 20, 20, 20)
        outputimage[:,:,:,0:10] = 1
        outputimage[:,:,:,10:20] = 0
        inputimage[:,0,:,:,10:20] = 1
        inputimage[:,1,:,:,0:10] = 1
        inputimage.requires_grad = True
        criterion = DiceLoss()
        loss = criterion(inputimage, outputimage)
        loss.backward()
        # Just checking that the autograd engine throws no errors


