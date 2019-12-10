# -*- coding: utf-8 -*-
"""
Job to visualize the network activations

@author: Karin van Garderen
"""
from glassimaging.execution.jobs.job import Job
import sys
import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from glassimaging.evaluation.evaluator import StandardEvaluator
from glassimaging.dataloading.brats18 import Brats18
from glassimaging.dataloading.btd import BTD
from glassimaging.dataloading.generic import GenericData
from glassimaging.evaluation.visualizer import NetworkVisualizer
from glassimaging.dataloading.transforms.randomcrop import RandomCrop
from glassimaging.dataloading.transforms.totensor import ToTensor
from glassimaging.dataloading.transforms.compose import Compose


class JobVisualize(Job):

    def __init__(self, configfile, name, tmpdir, homedir=None, uid=None):
        super().__init__(configfile, name, tmpdir, homedir, uid=uid)

    def getSchema(self):
        return {
            "type": "object",
            "properties": {
                "Nifti Source": {"type": "string"},
                "Model Source": {"type": "string"},
                "Patch size": {"type": "array"},
                "Batch size": {"type": "integer"},
                "Output type": {"type": "string"},
                "Dataset": {"type": "string"},
                "Number of patches": {"type": "integer"},
                "Splits from File": {"type": "string"},
                "Splits": {"type": "array"}
            },
            "required": ["Nifti Source", "Patch size", "Batch size", "Dataset", "Model Source"]
        }

    def run(self):
        ##### Create identifiers
        myconfig = self.config

        ##### Set network specifics
        patchsize = myconfig["Patch size"]
        batchsize = myconfig["Batch size"]

        #### Data specifics
        if "Splits" in myconfig:
            splits = myconfig["Splits"]
        else:
            splits = [0]

        ##### load datamanager
        loc = myconfig["Nifti Source"]
        if self.config['Dataset'] == 'Brats18':
            dataset = Brats18.fromFile(loc)
        elif self.config['Dataset'] == 'BTD':
            dataset = BTD.fromFile(loc)
        elif self.config['Dataset'] == 'Generic':
            dataset = GenericData.fromFile(loc)

        if "Splits from File" in myconfig:
            dataset.loadSplits(myconfig["Splits from File"])

        ##### Load model from source step
        sourcestep = myconfig["Model Source"]
        loc_model = os.path.join(self.datadir, sourcestep)
        config_model = self.loadConfig(os.path.join(loc_model, 'config.json'))
        sequences = config_model["Sequences"]

        transforms = [
            RandomCrop(output_size=tuple(patchsize)),
            ToTensor()
        ]
        transform = Compose(transforms)
        testset = dataset.getDataset(splits, sequences, transform=transform)
        dataloader = DataLoader(testset, batch_size=batchsize, num_workers=0, shuffle=True)

        evaluator = StandardEvaluator.loadFromCheckpoint(os.path.join(loc_model, 'model.pt'))
        self.logger.info('Dataloader has {n} images.'.format(n=len(testset)))

        visualizer = NetworkVisualizer(evaluator.net)

        for i_batch, sample_batched in enumerate(dataloader):
            if "Number of patches" in myconfig and i_batch == myconfig["Number of patches"]:
                break
            images = sample_batched["data"]

            #### Visualize
            acts = visualizer.getAllActivations(imagebatch=images)
            with open(os.path.join(self.tmpdir, 'activations.pickle'), 'wb') as f:
                pickle.dump(acts, f)

            images = images.detach().numpy()
            print(acts.shape)

            for i in range(acts.shape[0]):
            ### PLOT EACH CHANNEL ACTIVATIONS
                f = plt.figure(figsize=(20, 20))
                ax = f.add_subplot(2, 3, 1)
                ax.imshow(images[0, 0, 54, :, :], cmap='gray_r')
                ax = f.add_subplot(2, 3, 2)
                ax.imshow(images[0, 0, :, 54, :], cmap='gray_r')
                ax = f.add_subplot(2, 3, 3)
                ax.imshow(images[0, 0, :, :, 54], cmap='gray_r')

                ax = f.add_subplot(2, 3, 4)
                ax.imshow(acts[i, 54, :, :], cmap='inferno')
                ax = f.add_subplot(2, 3, 5)
                ax.imshow(acts[i, :, 54, :], cmap='inferno')
                ax = f.add_subplot(2, 3, 6)
                ax.imshow(acts[i, :, :, 54], cmap='inferno')

                f.savefig(os.path.join(self.tmpdir, 'activations_{}_{}.png'.format(i_batch, i)))
                f.close()

        self.tearDown()

def main(configfile, name, tmpdir, homedir=None):
    job = JobVisualize(configfile, name, tmpdir, homedir)
    job.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a job.')
    parser.add_argument('name', help='a name to call your job by.')
    parser.add_argument('configfile', help='path to a json config file.')
    parser.add_argument('tmpdir', help='directory for the output.')
    parser.add_argument('--log', help='additional directory to write logs to.')
    args = parser.parse_args()
    main(args.configfile, args.name, args.tmpdir, args.log)
