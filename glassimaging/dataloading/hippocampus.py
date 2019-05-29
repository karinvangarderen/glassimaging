# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
from glassimaging.dataloading.niftidataset import NiftiDataset
import logging
import json
from torch.utils.data import Dataset


class Hippocampus(NiftiDataset):

    available_sequences = ['t1']

    """The image paths for each subject are stored at initialization
    """
    def __init__(self, df=None):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df

    def importData(self, data_loc, nsplits = 4):
        images = glob.glob(os.path.join(data_loc, '*_T1_NuM_SubLR.nii.gz'))
        images = {os.path.basename(i).replace('_T1_NuM_SubLR.nii.gz', ''): i for i in images}
        patients = images.keys()

        segmentations = glob.glob(os.path.join(data_loc, '*_hippo_SubLR.nii.gz'))
        segmentations = {os.path.basename(i).replace('_hippo_SubLR.nii.gz', ''): i for i in segmentations}

        df = pd.DataFrame.from_dict(images, orient='index', columns=['t1'])
        for p in patients:
            df.at[p, 'seg'] = segmentations[p]

        self.df = df
        self.patients = patients

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """
    @staticmethod
    def fromFile(loc, nsplits = 4):
        instance = Hippocampus()
        instance.importData(loc, nsplits)
        logging.info('Hippocampus new dataloader created from ' + loc + '.')
        return instance

    def setSplits(self, splits_file):
        """ Load the information on cross-validation splits from a json file
        """
        with open(splits_file, 'r') as file:
            splits = json.load(file)
        # Set all patient to split -1, so that only patients in the actual splits file are included
        self.df['split'] = -1
        for i in range(0, len(splits)):
            for p in splits[i]:
                self.df.at[p, 'split'] = i

    def getDataset(self, splits=(), sequences = None, transform=None):
        if len(splits) == 0:
            splits = range(0, self.nsplits)
        if sequences is None:
            sequences = self.available_sequences
        dataset = HippocampusDataset(self.df.loc[[s in splits for s in self.df['split']]], sequences, transform=transform)
        return dataset


class HippocampusDataset(NiftiDataset, Dataset):

    def __init__(self, dataframe, sequences, transform=None):
        Dataset.__init__(self)
        NiftiDataset.__init__(self)
        self.df = dataframe
        self.sequences = sequences
        self.patients = self.df.index.values
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patientname = self.patients[idx]
        (image, segmentation) = self.loadSubjectImages(patientname, self.sequences)
        seg_file = self.getFileName(patientname, 'seg')
        sample = {'data': image, 'seg': segmentation, 'seg_file': seg_file, 'subject': patientname}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def saveListOfPatients(self, path):
        with open(path, 'w') as file:
            json.dump(self.patients.tolist(), file)
