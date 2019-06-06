import torch
import unittest
import numpy as np
from glassimaging.models.multipath import UNetMultipath
from glassimaging.models.ResUNet3D import ResUNet


class TestTorchnet(unittest.TestCase):

    # @unittest.skip("demonstrating skipping")
    def testRunModel(self):
        net = UNetMultipath(inputsize=4, outputsize=2, k=4)
        net.eval()
        inputshape = [64,64,64]
        inputimage = torch.randn(1,4,*inputshape)
        output = net(inputimage).detach().numpy()
        outputshape = list(output.shape[2:5])
        print(outputshape)

    def testLoadModels(self):
        basemodel = ResUNet(inputsize=1, outputsize=2, k=4)
        state_dict = basemodel.state_dict()
        for k in state_dict:
            print(k)
        net = UNetMultipath(inputsize=4, outputsize=2, k=4)
        list_of_dicts = [state_dict] * 4
        net.loadExistingModels(list_of_dicts)
        inputshape = [64,64,64]
        inputimage = torch.randn(1,4,*inputshape)
        output = net(inputimage).detach().numpy()

