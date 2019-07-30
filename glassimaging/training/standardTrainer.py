# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import sys
from datetime import datetime as dt
import os
from glassimaging.models.utils import createModel


class StandardTrainer:
    lr = 0.0001
    momentum = 0.9

    def __init__(self, net, optimizer, scheduler=None):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.DEBUG)
        self.net = net
        self.device = StandardTrainer.getDevice()
        if torch.cuda.device_count() == 2:
            self.logger.info('Two devices being used.')
            self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1]).cuda()
            self.parallel = True
        else:
            self.parallel = False
            self.logger.info('Number of devices being used: {n}'.format(n=torch.cuda.device_count()))
            self.net = self.net.to(self.device)
        torch.backends.cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=3)
        else:
            self.scheduler = scheduler
        self.batchLog = pd.DataFrame(columns=['Epoch', 'DateTime', 'LR', 'Momentum', 'Loss', 'Train or Test'])

    def setLogger(self, logger):
        self.logger = logger


    def setLossFunction(self, function):
        self.criterion = function

    @staticmethod
    def getDevice():
       return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def initFromDesc(desc):
        net = createModel(desc)
        optimizer = optim.Adam(net.parameters(), lr=StandardTrainer.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=3)
        return StandardTrainer(net, optimizer, scheduler=scheduler)

    def saveModel(self, loc):
        if self.parallel:
            model_desc = self.net.module.getDesc()
            state_dict = self.net.module.state_dict()
        else:
            model_desc = self.net.getDesc()
            state_dict = self.net.state_dict()
        torch.save({
            'model_desc': model_desc,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            }, loc)

    def getModelDesc(self):
        if self.parallel:
            return self.net.module.getDesc()
        else:
            return self.net.getDesc()

    def trainWithBatch(self, imagebatch, targetbatch):
        self.net = self.net.train()
        targetbatch = targetbatch.long()
        targetbatch = targetbatch[:,:,:,:]
        imagebatch = imagebatch.float()
        targetbatch = targetbatch.to(self.device)
        imagebatch = imagebatch.to(self.device)
        output = self.net(imagebatch)
        loss = self.criterion(output, targetbatch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def lossWithBatch(self, imagebatch, targetbatch):
        self.net = self.net.eval()
        targetbatch = targetbatch.long()
        targetbatch = targetbatch[:,:,:,:]
        imagebatch = imagebatch.float()
        targetbatch = targetbatch.to(self.device)
        imagebatch = imagebatch.to(self.device)
        with torch.no_grad():
            output = self.net(imagebatch)
            loss = self.criterion(output, targetbatch)
        return loss.detach()

    def trainWithLoader(self, torchloader, epochs, testloader=None, maxBatchesPerEpoch=None, testinterval=5):
        self.logger.info('Using cuda version {v}.'.format(v=torch.version.cuda))
        for e in range(0, epochs):
            self.logger.info('Epoch {e} started training.'.format(e=e))
            running_loss = 0.0
            num_items = 0
            for i_batch, sample_batched in enumerate(torchloader):
                img = sample_batched["data"]
                seg = sample_batched["seg"]
                loss = self.trainWithBatch(img, seg)
                running_loss += loss.item()
                num_items += img.size(0)
                if maxBatchesPerEpoch is not None and i_batch >= maxBatchesPerEpoch - 1:
                    break
            loss = running_loss / (i_batch + 1)
            self.scheduler.step(loss)
            self.noteLoss(loss, category="DuringTraining", epoch=e)
            if e % testinterval == 0:
                loss = self.getLossWithLoader(torchloader, maxBatches=maxBatchesPerEpoch)
                self.noteLoss(loss, category="Train", epoch=e)
                if testloader is not None:
                    testloss = self.getLossWithLoader(testloader, maxBatches=maxBatchesPerEpoch)
                    self.noteLoss(testloss, category="Test", epoch=e)
                    self.logger.info('Loss calculated for epoch {e}. Train loss is {loss}.'.format(e=e, loss=loss))

    def noteLoss(self, loss, category='DuringTraining', loader=0, epoch=0):
        self.batchLog = self.batchLog.append([{'Epoch': epoch, 'DateTime': dt.now(), 'LR': self.lr,
                                               'Momentum': self.momentum, 'Loss': loss, 'Train or Test': category,
                                               'Loader': loader}], ignore_index=True)

    def getLossWithLoader(self, torchloader, maxBatches=None):
        running_loss = 0.0
        num_items = 0
        for i_batch, sample_batched in enumerate(torchloader):
            img = sample_batched["data"]
            seg = sample_batched["seg"]
            loss = self.lossWithBatch(img, seg)
            running_loss += loss.item()
            num_items += img.size(0)
            if maxBatches != None and i_batch >= maxBatches:
                    break
        return running_loss/(i_batch + 1)

    def writeLog(self, path):
        self.batchLog.to_csv(os.path.join(path, 'trainlog.csv'))
        train = self.batchLog.loc[self.batchLog['Train or Test'] == 'Train']
        test = self.batchLog.loc[self.batchLog['Train or Test'] == 'Test']
        duringTraining = self.batchLog.loc[self.batchLog['Train or Test'] == 'DuringTraining']
        plt.plot(test['Epoch'], test['Loss'], label = 'Test')
        plt.plot(train['Epoch'], train['Loss'], label='Train')
        plt.plot(duringTraining['Epoch'], duringTraining['Loss'], label='Loss while training')
        plt.legend()
        plt.savefig(os.path.join(path, 'trainlog.png'))
        plt.close()

    @staticmethod
    def loadFromCheckpoint(loc):
        checkpoint = torch.load(loc, map_location=StandardTrainer.getDevice())
        net = createModel(checkpoint['model_desc'])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(net.parameters(), lr=StandardTrainer.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=3)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return StandardTrainer(net, optimizer, scheduler)

    def inferWithImage(self, image):
        self.net = self.net.eval()
        with torch.no_grad():
            torchimage = image.float()
            torchimage = torchimage.to(self.device)
            output = self.net(torchimage)
            output = output.detach()
        return output

