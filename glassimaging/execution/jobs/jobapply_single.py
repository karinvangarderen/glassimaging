# -*- coding: utf-8 -*-
"""
Job to apply a network to a single scan

@author: Karin van Garderen
"""
import os
from glassimaging.execution.jobs.job import Job
from torch.utils.data import DataLoader
from glassimaging.evaluation.evaluator import StandardEvaluator
from glassimaging.dataloading.generic import SingleInstanceDataset
from glassimaging.dataloading.transforms.totensor import ToTensor
from glassimaging.evaluation.utils import logDataLoader


class JobApplySingle(Job):

    def __init__(self, t1, t2, flair, t1gd, modelpath, brainmaskpath, configfile, outdir = '.'):
        super().__init__(configfile, 'result', outdir)
        self.config["Nifti paths"] = [t1, t2, flair, t1gd]
        self.config["Model path"] = modelpath
        self.config["Brainmask path"] = brainmaskpath

    def getSchema(self):
        return {
            "type": "object",
            "properties": {
                "Nifti paths": {"type": "array"},
                "Brainmask path": {"type": "string"},
                "Model path": {"type": "string"},
                "Patch size": {"type": "array"}
            },
            "required": ["Patch size"]
        }

    def run(self):
        ##### Create identifiers
        myconfig = self.config

        ##### Set network specifics
        patchsize = myconfig["Patch size"]

        ##### Load model from source step
        loc_model = myconfig["Model path"]

        dataset = SingleInstanceDataset(myconfig['Nifti paths'], brainmask_path=myconfig['Brainmask path'], transform= ToTensor())
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

        logDataLoader(dataloader, self.tmpdir)
        evaluator = StandardEvaluator.loadFromCheckpoint(loc_model)

        sample_batched = next(iter(dataloader))
        images = sample_batched["data"]
        header_sources = sample_batched["header_source"]
        resultpaths = [os.path.join(self.tmpdir, 'segmentation.nii.gz')]
        evaluator.segmentNifti(images, header_sources, patchsize, resultpaths)
        self.logger.info('Nifti image segmented for.')
        self.tearDown()

def runWithInput(self, t1, t2, flair, t1gd, modelpath, outputpath, brainmaskpath):
    self.config["Nifti paths"] = [t1, t2, flair, t1gd]
    self.config["Model path"] = modelpath
    self.config["Output path"] = outputpath
    self.config["Brainmask path"] = brainmaskpath
    self.run()

def main(configfile, name, tmpdir, homedir=None):
    job = JobApplySingle(configfile, name, tmpdir, homedir)
    job.run()


