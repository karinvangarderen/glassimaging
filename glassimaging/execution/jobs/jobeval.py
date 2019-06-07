# -*- coding: utf-8 -*-
"""
Job to evaluate a network

@author: Karin van Garderen
"""
from glassimaging.execution.jobs.job import Job
import sys
import os
from torch.utils.data import DataLoader
from glassimaging.evaluation.evaluator import StandardEvaluator
from glassimaging.evaluation.utils import segmentNifti, plotResultImage, getPerformanceMeasures
from glassimaging.dataloading.brats18 import Brats18
from glassimaging.dataloading.btd import BTD
from glassimaging.dataloading.hippocampus import Hippocampus
from glassimaging.dataloading.transforms.totensor import ToTensor
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

class JobEval(Job):
    
    def __init__(self, configfile, name,  tmpdir, homedir = None, uid = None):
        super().__init__(configfile, name,  tmpdir, homedir, uid = uid)
        
    def run(self):
        ##### Create identifiers
        myconfig = self.config
        
        ##### Set network specifics
        patchsize = myconfig["Patch size"]
        batchsize = myconfig["Batch size"]
        output_type = myconfig["Output type"]
        only_first = myconfig["Only first"]
    
        #### Data specifics
        splits = myconfig["Splits"]
        
        ##### load datamanager
        loc = myconfig["Nifti Source"]
        if self.config['Dataset'] == 'Brats18':
            dataset = Brats18.fromFile(loc)
        elif self.config['Dataset'] == 'BTD':
            dataset = BTD.fromFile(loc)
        elif self.config['Dataset'] == 'Hippocampus':
            dataset = Hippocampus.fromFile(loc)
        
        if "Splits from File" in myconfig:
            dataset.loadSplits(myconfig["Splits from File"])

        
        ##### Load model from source step
        sourcestep = myconfig["Model Source"]
        loc_model = os.path.join(self.datadir, sourcestep)
        config_model = self.loadConfig(os.path.join(loc_model, 'config.json'))

        if "Sequences" in self.config:
            sequences = self.config["Sequences"]
        else:
            sequences = config_model["Sequences"]

        transform = ToTensor()
        if 'Target' in self.config and self.config['Target'] == 'Brainmask':
            testset = dataset.getBrainmaskDataset(splits, sequences, transform=transform)
        else:
            testset = dataset.getDataset(splits, sequences, transform=transform)
        dataloader = DataLoader(testset, batch_size=batchsize, num_workers=0, shuffle=True)

        evaluator = StandardEvaluator.loadFromCheckpoint(os.path.join(loc_model, 'model.pt'))
        self.logger.info('Dataloader has {n} images.'.format(n=len(testset)))
        all_dice = []
        all_dice_core = []
        all_dice_enhancing = []
        results = pd.DataFrame(columns = ['sample', 'subject', 'class', 'TP', 'FP', 'FN', 'TN', 'dice'])
        for i_batch, sample_batched in enumerate(dataloader):
            ######### TODO: work with original dataset
            images = sample_batched['data']
            segfiles = sample_batched['seg_file']
            subjects = sample_batched['subject']
            resultpaths = [os.path.join(self.tmpdir, s+'_segmented.nii.gz') for s in subjects]
            classifications = evaluator.segmentNifti(images, segfiles, patchsize, resultpaths)
            for i in range(0, len(subjects)):
                seg = nib.load(segfiles[i]).get_fdata()
                plotResultImage(dataset, resultpaths[i], self.tmpdir, subjects[i], output_type = output_type)
                for c in range(0,5):
                    truth = seg == c
                    positive = classifications[i] == c
                    (dice, TT, FP, FN, TN) = getPerformanceMeasures(positive, truth)
                    results = results.append({'sample':i_batch, 'class':c, 'subject':subjects[i], 'TP': TT, 'FP': FP, 'FN': FN, 'TN': TN, 'dice': dice}, ignore_index=True)
                    if c == 4:
                        all_dice_enhancing.append(dice)
                class_whole = classifications[i] > 0
                result_core = (classifications[i] == 1) | (classifications[i] == 4)
                truth_whole = seg > 0
                truth_core = (seg == 1) | (seg == 4)
                (dice, TT, FP, FN, TN) = getPerformanceMeasures(class_whole, truth_whole)
                (dice_core, TT_core, FP_core, FN_core, TN_core) = getPerformanceMeasures(result_core, truth_core)
                all_dice.append(dice)
                all_dice_core.append(dice_core)
                self.logger.info('Nifti image segmented for ' + subjects[i] + '. Dice: ' + str(dice))
                results = results.append({'sample':i_batch, 'class':'whole', 'subject':subjects[i], 'TP': TT, 'FP': FP, 'FN': FN,'TN': TN, 'dice': dice}, ignore_index=True)
                results = results.append(
                    {'sample': i_batch, 'class': 'core', 'subject': subjects[i], 'TP': TT_core, 'FP': FP_core, 'FN': FN_core,
                     'TN': TN_core, 'dice': dice_core}, ignore_index=True)
            if only_first: break
        dice_mean = sum(all_dice)/len(all_dice)
        dice_core = sum(all_dice_core)/len(all_dice_core)
        dice_enhancing = sum(all_dice_enhancing) / len(all_dice_enhancing)
        plt.boxplot(all_dice)
        plt.savefig(os.path.join(self.tmpdir, 'boxplot_dice.png'))
        plt.close()
        results.to_csv(os.path.join(self.tmpdir, 'results_eval.csv'))
        dataset.saveSplits(self.tmpdir)
        self.logger.info('evaluation finished. Dice coefficient: whole: {}, core: {}, enhancing: {}'.format(dice_mean, dice_core, dice_enhancing))
        
        self.tearDown()


if __name__ == '__main__':
    name = sys.argv[1]
    configfile = sys.argv[2]
    tmpdir = sys.argv[3]
    homedir = sys.argv[4]
    job = JobEval(configfile, name, tmpdir, homedir)
    job.run()
