# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
from glassimaging.dataloading.niftidataset import NiftiDataset
import logging
import json
from torch.utils.data import Dataset


class Brats18(NiftiDataset):

    available_sequences = ['flair', 't1', 't1Gd', 't2']

    """The image paths for each subject are stored at initialization
    """
    def __init__(self, df=None):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df

    def importData(self, data_loc, nsplits = 5):
        flair_images = glob.glob(os.path.join(data_loc, '*', '*_flair.nii.gz'))
        flair_images = {os.path.basename(os.path.dirname(i)): i for i in flair_images}
        patients = flair_images.keys()

        t1_images = glob.glob(os.path.join(data_loc, '*', '*_t1.nii.gz'))
        t1_images = {os.path.basename(os.path.dirname(i)): i for i in t1_images}

        t1Gd_images = glob.glob(os.path.join(data_loc, '*', '*_t1ce.nii.gz'))
        t1Gd_images = {os.path.basename(os.path.dirname(i)): i for i in t1Gd_images}

        t2_images = glob.glob(os.path.join(data_loc, '*', '*_t2.nii.gz'))
        t2_images = {os.path.basename(os.path.dirname(i)): i for i in t2_images}

        segmentations = glob.glob(os.path.join(data_loc, '*', '*_seg.nii.gz'))
        segmentations = {os.path.basename(os.path.dirname(i)): i for i in segmentations}

        df = pd.DataFrame.from_dict(flair_images, orient='index', columns=['flair'])
        df['t1'] = ''
        df['t1Gd'] = ''
        df['t2'] = ''
        df['seg'] = ''
        for p in patients:
            df.at[p, 't1'] = t1_images[p]
            df.at[p, 't1Gd'] = t1Gd_images[p]
            df.at[p, 't2'] = t2_images[p]
            df.at[p, 'seg'] = segmentations[p]

        self.df = df
        self.patients = patients

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """
    @staticmethod
    def fromFile(loc, nsplits = 5):
        instance = Brats18()
        instance.importData(loc, nsplits)
        logging.info('Brats new Datamanager created from ' + loc + '.')
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
        dataset = Brats18Dataset(self.df.loc[[s in splits for s in self.df['split']]], sequences, transform=transform)
        return dataset


class Brats18Dataset(NiftiDataset, Dataset):

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
