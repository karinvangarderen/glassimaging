# -*- coding: utf-8 -*-
"""
Job to train a network

@author: Karin van Garderen
"""

import sys
import os

from torch.utils.data import DataLoader
from glassimaging.execution.jobs.job import Job
from glassimaging.training.multipathTrainer import MultipathTrainer
from glassimaging.dataloading.brats18 import Brats18
from glassimaging.dataloading.btd import BTD
from glassimaging.dataloading.hippocampus import Hippocampus
from glassimaging.dataloading.transforms.randomcrop import RandomCrop
from glassimaging.dataloading.transforms.totensor import ToTensor
from glassimaging.dataloading.transforms.compose import Compose
from glassimaging.evaluation.utils import logDataLoader

class JobTrainMultipath(Job):
    
    def __init__(self, configfile, name,  tmpdir, homedir=None, uid=None):
        super().__init__(configfile, name,  tmpdir, homedir, uid=uid)

    def getDataloader(self):
        batchsize = self.config['Batch size']
        loc = os.path.join(self.datadir, self.config["Nifti Source"])
        sequences = self.config["Sequences"] if "Sequences" in self.config else None
        if self.config['Dataset'] == 'Brats18':
            dataset = Brats18.fromFile(loc)
        elif self.config['Dataset'] == 'BTD':
            dataset = BTD.fromFile(loc)
        elif self.config['Dataset'] == 'Hippocampus':
            dataset = Hippocampus.fromFile(loc)

        if "Splits from File" in self.config:
            dataset.setSplits(self.config["Splits from File"])
        elif "Crossval Splits" in self.config:
            dataset.createCVSplits(self.config["Crossval Splits"])
        #### Data specifics
        splits = self.config['Splits']
        testsplits = self.config['Testsplits']
        dataset.saveSplits(self.tmpdir)
        targetsize = tuple(self.config["Patch size"])
        imgsize = targetsize
        transform = Compose([
            RandomCrop(output_size=imgsize),
            ToTensor()
        ])

        if 'Target' in self.config and self.config['Target'] == 'Brainmask':
            trainset = dataset.getBrainmaskDataset(splits, sequences, transform=transform)
        else:
            trainset = dataset.getDataset(splits, sequences, transform=transform)

        trainset.saveListOfPatients(os.path.join(self.tmpdir, 'trainset.json'))
        self.logger.info('Generating patches with input size ' + str(imgsize) + ' and outputsize ' + str(targetsize))

        n_workers = self.config['Num Workers']
        trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=n_workers, shuffle=True)
        if len(testsplits) > 0:
            if 'Target' in self.config and self.config['Target'] == 'Brainmask':
                testset = dataset.getBrainmaskDataset(testsplits, sequences, transform=transform)
            else:
                testset = dataset.getDataset(testsplits, sequences, transform=transform)
            testloader = DataLoader(testset, batch_size=batchsize, num_workers=n_workers, shuffle=True)
        else:
            testloader = None

        logDataLoader(trainloader, self.tmpdir)

        return trainloader, testloader
        
    def run(self):
        myconfig = self.config

        epochs = myconfig['Epochs']
        maxBatchesPerEpoch = myconfig["Batches per epoch"]
        
        ##### Load model from source step
        sourcestep = myconfig["Model Source"]
        model_loc = os.path.join(self.datadir, sourcestep)
        trainer = MultipathTrainer.loadFromCheckpoint(os.path.join(model_loc, 'model.pt'))
        trainer.setLogger(self.logger)
        model_desc = trainer.getModelDesc()
        model = model_desc[0]
        self.logger.info('model loaded from ' + model_loc + '.')


        if "Model Sources" in myconfig:
            sourcesteps = myconfig["Model Sources"]
            model_loc = [os.path.join(self.datadir, sourcesteps[i]) for i in range(0,4)]
            network_locations = [os.path.join(l, 'model.pt') for l in model_loc]
            sourceconfigs = [self.loadConfig(os.path.join(l, 'config.json')) for l in model_loc]
            sequences = [conf["Sequences"][0] for conf in sourceconfigs]
            self.config['Sequences'] = sequences
            trainer.loadExistingModels(network_locations)
            self.logger.info('Models loaded into paths')

        (trainloader, testloader) = self.getDataloader()
        if myconfig['Freeze UNets']:
            trainer.trainLastLayerOnly()
            self.logger.info('Only training the final layers')

        trainer.trainWithLoader(trainloader, epochs, testloader=testloader, maxBatchesPerEpoch=maxBatchesPerEpoch)
        self.logger.info('training finished for ' + str(epochs) + ' epochs.')
        trainer.writeLog(self.tmpdir)
        modelpath = os.path.join(self.tmpdir, 'model.pt')
        trainer.saveModel(modelpath)
        self.logger.info('model and training log saved under ' + modelpath + '.')
        del trainer
        self.tearDown()

        

if __name__ == '__main__':
    name = sys.argv[1]
    configfile = sys.argv[2]
    tmpdir = sys.argv[3]
    homedir = sys.argv[4]
    job = JobTrainMultipath(configfile, name, tmpdir, homedir)
    job.run()
