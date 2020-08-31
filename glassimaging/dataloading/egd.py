# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
from glassimaging.dataloading.niftidataset import NiftiDataset
import logging
import json
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class EGD(NiftiDataset):

    available_sequences = ['flair']
    """The image paths for each subject are stored at initialization
    """
    def __init__(self, df=None, sequences=('flair')):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df
        self.sequences = sequences

    def importData(self, data_loc, nsplits=5):
        subjects = os.listdir(data_loc)
        self.loc = data_loc

        df = pd.DataFrame(columns=['subject', 'flair','seg'])

        for s in subjects:
            flair = os.path.join(data_loc, s, 'flair.nii.gz')
            seg = os.path.join(data_loc, s, 'mask.nii.gz')
            if os.path.exists(flair) and os.path.exists(seg):

                df = df.append({'subject': s, 'flair': flair,
                                'seg': seg, 'split': 0}, ignore_index=True)
        self.df = df.set_index('subject')
        self.patients = self.df.index.values

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """
    @staticmethod
    def fromFile(loc, nsplits = 5):
        instance = EGD()
        instance.importData(loc, nsplits=nsplits)
        logging.info('EGD new Datamanager created from ' + loc + '.')
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
        dataset = EGDDataset(self.df.loc[[s in splits for s in self.df['split']]], sequences,
                             transform=transform)
        return dataset

class EGDDataset(NiftiDataset, Dataset):

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
        (image, segmentation) = self.loadSubjectImages(patientname, self.sequences, normalized=False)
        for i in range(0, image.shape[0]):
            img = image[i]
            maxval = np.percentile(img, 99)
            minval = np.percentile(img, 1)
            img = np.clip(img, minval, maxval)
            mean = np.mean(img)
            std = np.std(img)
            if std == 0:
                raise ValueError('Standard deviation of image is zero')
            img = (img - mean) / std
            image[i] = img
        segmentation = (segmentation > 0).astype(int)
        segfile = self.df.loc[patientname]['seg']
        sample = {'data': image, 'seg': segmentation, 'seg_file': segfile, 'subject': patientname}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def saveListOfPatients(self, path):
        with open(path, 'w') as file:
            json.dump(self.patients.tolist(), file)

