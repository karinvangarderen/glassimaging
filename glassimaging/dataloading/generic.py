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


class GenericData(NiftiDataset):

    available_sequences = ['flair', 't1', 't1Gd', 't2']
    SEQ_TRANSLATION = {'t1Gd': 't1gd.nii.gz', 't2': 't2.nii.gz', 'flair': 'flair.nii.gz', 't1': 't1.nii.gz'}
    """The image paths for each subject are stored at initialization
    """
    def __init__(self, df=None, brainmask=True, segmentation=True, sequences=('t1', 'flair', 't2', 't1Gd')):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df
        self.brainmask = brainmask
        self.segmentation = segmentation
        self.sequences = sequences

    def importData(self, data_loc, nsplits=1):
        subjects = os.listdir(data_loc)
        self.loc = data_loc

        df = pd.DataFrame(columns=['subject', 't1Gd', 't2', 'flair', 't1', 'brainmask'])

        for s in subjects:
            list_t1gd = glob.glob(os.path.join(data_loc, s, self.SEQ_TRANSLATION['t1Gd']))
            list_t1gd = [f for f in list_t1gd if "mask" not in f]
            list_t2 = glob.glob(os.path.join(data_loc, s, self.SEQ_TRANSLATION['t2']))
            list_t2 = [f for f in list_t2 if "mask" not in f]
            list_flair = glob.glob(os.path.join(data_loc, s, self.SEQ_TRANSLATION['flair']))
            list_flair = [f for f in list_flair if "mask" not in f]
            list_t1 = glob.glob(os.path.join(data_loc, s, self.SEQ_TRANSLATION['t1']))
            list_t1 = [f for f in list_t1 if "mask" not in f]
            list_brainmask = glob.glob(os.path.join(data_loc, s, 'brainmask.nii.gz'))

            if len(list_t1gd) >= 1:
                t1gd = list_t1gd[0]
            else:
                t1gd = ''
            if len(list_t2) >= 1:
                t2 = list_t2[0]
            else:
                t2 = ''
            if len(list_flair) >= 1:
                flair = list_flair[0]
            else:
                flair = ''
            if len(list_t1) >= 1:
                t1 = list_t1[0]
            else:
                t1 = ''
            if len(list_brainmask) >= 1:
                brainmask = list_brainmask[0]
            df = df.append({'subject': s, 't1Gd': t1gd, 't2': t2, 't1': t1, 'flair': flair, 'brainmask': brainmask,
                            'split': 0}, ignore_index=True)
        self.df = df.set_index('subject')
        for seq in self.sequences:
            self.df = self.df.loc[self.df[seq] != '']
        self.patients = self.df.index.values

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """
    @staticmethod
    def fromFile(loc, nsplits = 1, brainmask=True, sequences=('t1', 'flair', 't2', 't1Gd')):
        instance = GenericData(nsplits, brainmask,  sequences)
        instance.importData(loc, nsplits)
        logging.info('Data loaded created from ' + loc + '.')
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
        dataset = GenericDataset(self.df.loc[[s in splits for s in self.df['split']]], sequences,
                             transform=transform, brainmask=self.brainmask)
        return dataset


class GenericDataset(NiftiDataset, Dataset):

    def __init__(self, dataframe, sequences, transform=None, brainmask=True):
        Dataset.__init__(self)
        NiftiDataset.__init__(self)
        self.brainmask = brainmask
        self.df = dataframe
        self.sequences = sequences
        self.patients = self.df.index.values
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patientname = self.patients[idx]

        image = self.loadSubjectImagesWithoutSeg(patientname, self.sequences, normalized=False)
        if self.brainmask:
            brainmask = self.loadSegBinarize(self.df.loc[patientname]['brainmask'])

        for i in range(0, image.shape[0]):
            img = image[i]
            maxval = np.percentile(img, 99)
            minval = np.percentile(img, 1)
            img = np.clip(img, minval, maxval)
            if self.brainmask:
                values_nonzero = img[np.nonzero(brainmask)]
                mean_nonzero = np.mean(values_nonzero)
                std_nonzero = np.std(values_nonzero)
                if std_nonzero == 0:
                    raise ValueError('Standard deviation of image is zero')
                img = (img - mean_nonzero) / std_nonzero
                img[np.where(brainmask == 0)] = 0
            else:
                mean = np.mean(img)
                std = np.std(img)
                if std == 0:
                    raise ValueError('Standard deviation of image is zero')
                img = (img - mean) / std
            image[i] = img

        sample = {'data': image, 'subject': patientname, 'header_source': self.df.loc[patientname]['brainmask'], 't1_source': self.df.loc[patientname]['t1']}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def saveListOfPatients(self, path):
        with open(path, 'w') as file:
            json.dump(self.patients.tolist(), file)
