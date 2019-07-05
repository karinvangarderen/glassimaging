import torch
import torch.nn as nn
import collections
from glassimaging.models.ResUNet3D import ResUNetBody


class UNetMultipath(nn.Module):

    def __init__(self, inputsize=4, outputsize=2, k=8, p_drop=0.125):
        super(UNetMultipath, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.k = k
        self.unets = nn.ModuleList(
            [ResUNetBody(inputsize=1, outputsize=outputsize, k=k) for _ in range(0, inputsize)])
        self.p_drop = p_drop
        self.fullyConnected = nn.Conv3d(inputsize * k * 2, k * 4, 1, padding=0)
        self.fullyConnected2 = nn.Conv3d(k * 4, k * 4, 1, padding=0)
        self.votingLayer = nn.Conv3d(k * 4, outputsize, 1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.inputAvailable = [True] * inputsize
        self.p = torch.tensor([1 - self.p_drop] * self.inputsize)

    def getRandomAvailable(self):
        res = torch.bernoulli(self.p)
        while not torch.sum(res) > 0:
            res = torch.bernoulli(self.p)
        return res

    def setPDrop(self, p_drop):
        self.p_drop = p_drop
        self.p = self.p.new_tensor([1 - self.p_drop] * self.inputsize)

    def loadExistingModels(self, list_of_statedicts):
        for i in range(0, len(self.unets)):
            ## we need only the 'body'
            state_dict = list_of_statedicts[i]
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                body = k[0:5]
                if body == 'body.':
                    name = k[5:]  # strip body.
                    new_state_dict[name] = v
            ## Check that we correctly selected the keys by settting strict=True
            self.unets[i].load_state_dict(new_state_dict, strict=True)

    def trainLastLayerOnly(self):
        for net in self.unets:
            for p in net.parameters():
                p.requires_grad_(False)

    def getLastLayer(self):
        return self.fullyConnected

    def setInputAvailability(self, avail):
        self.inputAvailable = avail

    def forward(self, x):
        output = []

        if self.training:
            input_avail = self.getRandomAvailable()
        else:
            input_avail = self.getRandomAvailable()
            input_avail = input_avail.new_tensor(self.inputAvailable)
        factor = self.inputsize / torch.sum(input_avail)
        factor = factor.cuda(x.get_device()) if x.is_cuda else factor
        for i in range(0, self.inputsize):
            input_tensor = x[:, i:i + 1, :, :, :].clone()
            channel = self.unets[i](input_tensor)
            if input_avail[i]:
                channel = channel * factor
            else:
                channel = channel.new_zeros(channel.size(), requires_grad=False)
            output.append(channel)
        output = torch.cat(output, dim=1)
        res = self.fullyConnected(output)
        res = self.fullyConnected2(res)
        res = self.votingLayer(res)
        res = self.softmax(res)
        return res

    @staticmethod
    def initFromDesc(desc):
        inputsize = desc['inputsize']
        outputsize = desc['outputsize']
        k = desc['k'] if 'k' in desc else 16
        p_drop = desc['p_drop'] if 'p_drop' in desc else 0.125
        return UNetMultipath(outputsize=outputsize, inputsize=inputsize, k=k, p_drop=p_drop)

    def getDesc(self):
        return ['multipath', {
            'inputsize': self.inputsize,
            'outputsize': self.outputsize,
            'p_drop': self.p_drop,
            'k': self.k
        }]

class UNetSharedRep(nn.Module):


    def getDesc(self):
        return ['sharedrep', {
            'inputsize': self.inputsize,
            'outputsize': self.outputsize,
            'p_drop': self.p_drop,
            'k': self.k
        }]

    @staticmethod
    def initFromDesc(desc):
        inputsize = desc['inputsize']
        outputsize = desc['outputsize']
        k = desc['k'] if 'k' in desc else 16
        p_drop = desc['p_drop'] if 'p_drop' in desc else 0.125
        return UNetSharedRep(outputsize=outputsize, inputsize=inputsize, k=k, p_drop=p_drop)

    def __init__(self, inputsize=4, outputsize=2, k=8, p_drop=0.125):
        super(UNetSharedRep, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.k = k
        self.unets = nn.ModuleList(
            [ResUNetBody(inputsize=1, outputsize=outputsize, k=k) for _ in range(0, inputsize)])
        self.p_drop = p_drop
        self.fullyConnected = nn.Conv3d(k * 4, k * 4, 1, padding=0)
        self.fullyConnected2 = nn.Conv3d(k * 4, k * 4, 1, padding=0)
        self.votingLayer = nn.Conv3d(k * 4, outputsize, 1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.inputAvailable = [True] * inputsize
        self.p = torch.tensor([1 - self.p_drop] * self.inputsize)


    def getRandomAvailable(self):
        res = torch.bernoulli(self.p)
        while not torch.sum(res) > 0:
            res = torch.bernoulli(self.p)
        return res

    def setPDrop(self, p_drop):
        self.p_drop = p_drop
        self.p = self.p.new_tensor([1 - self.p_drop] * self.inputsize)

    def loadExistingModels(self, list_of_statedicts):
        for i in range(0, len(self.unets)):
            ## we need only the 'body'
            state_dict = list_of_statedicts[i]
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                body = k[0:5]
                if body == 'body.':
                    name = k[5:]  # strip body.
                    new_state_dict[name] = v
            ## Check that we correctly selected the keys by settting strict=True
            self.unets[i].load_state_dict(new_state_dict, strict=True)

    def trainLastLayerOnly(self):
        for net in self.unets:
            for p in net.parameters():
                p.requires_grad_(False)

    def getLastLayer(self):
        return self.fullyConnected

    def setInputAvailability(self, avail):
        self.inputAvailable = avail

    def forward(self, x):
        output = []

        if self.training:
            input_avail = self.getRandomAvailable()
        else:
            input_avail = self.getRandomAvailable()
            input_avail = input_avail.new_tensor(self.inputAvailable)
        factor = self.inputsize / torch.sum(input_avail)
        factor = factor.cuda(x.get_device()) if x.is_cuda else factor
        for i in range(0, self.inputsize):
            input_tensor = x[:, i:i + 1, :, :, :].clone()
            channel = self.unets[i](input_tensor)
            if input_avail[i]:
                channel = channel * factor
            else:
                channel = channel.new_zeros(channel.size(), requires_grad=False)
            output.append(channel)
        output = torch.stack(output, dim=0)
        mean = torch.mean(output, 0)
        if len(output) > 1:
            var = torch.var(output, 0)
        else:
            var = mean.new_zeros(mean.shape)
        output = torch.cat([mean, var], dim=1)
        res = self.fullyConnected(output)
        res = self.fullyConnected2(res)
        res = self.votingLayer(res)
        res = self.softmax(res)
        return res
