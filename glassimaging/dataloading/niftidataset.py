# -*- coding: utf-8 -*-

import os
import pandas as pd
import nibabel as nib
import numpy as np
import json

class NiftiDataset:

    def __init__(self):
        self.df = pd.DataFrame()

    """Return an image based on the path as a numpy array

    Not normalized
    """
    def loadImage(self, path):
        img = nib.load(path)
        img = img.get_fdata()
        return img

    def __len__(self):
        return len(self.df.index)

    """Return an image based on the path as a numpy array

    Normalized using the z-score
    """

    def loadImageNormalized(self, path):
        img = nib.load(path)
        img = img.get_fdata()
        values_nonzero = img[np.nonzero(img)]
        mean_nonzero = np.mean(values_nonzero)
        std_nonzero = np.std(values_nonzero)
        if std_nonzero == 0:
            raise ValueError('Standard deviation of image is zero')
        img[np.nonzero(img)] = (img[np.nonzero(img)] - mean_nonzero) / std_nonzero
        return img

    """Return a segmentation based on the path as boolean numpy array
    """
    def loadSegBinarize(self, path):
        img = nib.load(path)
        return img.get_fdata() > 0.5

    def loadSeg(self, path):
        img = nib.load(path)
        return img.get_fdata()


    """Return the full stacked sequence of images of a subject

    Useful for evaluation
    """
    def loadSubjectImages(self, subject, sequences, normalized=True):
        if normalized:
            image = [self.loadImageNormalized(self.df.loc[subject][seq]) for seq in sequences]
        else:
            image = [self.loadImage(self.df.loc[subject][seq]) for seq in sequences]
        image = np.stack(image)
        segmentation = self.loadSeg(self.df.loc[subject]['seg'])
        return image, segmentation

    """Return the full stacked sequence of images of a subject

    Useful for evaluation
    """
    def loadSubjectImagesWithoutSeg(self, subject, sequences, normalized=True):
        if normalized:
            image = [self.loadImageNormalized(self.df.loc[subject][seq]) for seq in sequences]
        else:
            image = [self.loadImage(self.df.loc[subject][seq]) for seq in sequences]
        image = np.stack(image)
        return image

    def createCVSplits(self, nsplits):
        self.df['split'] = -1
        split = -1
        for p in np.random.permutation(list(self.df.index)):
            split = (split + 1) % nsplits
            self.df.at[p, 'split'] = split
        self.nsplits = nsplits
        return

    def loadSplits(self, path):
        with open(path, 'r') as file:
            splits = json.load(file)
        ######## Set all patient to split -1, so that only patients in the actual splits file are included
        self.df['split'] = -1
        for i in range(0, len(splits)):
            for p in splits[i]:
                self.df.at[p, 'split'] = i

    def saveSplits(self, loc):
        splits = self.df.split.unique()
        d = [None] * len(splits)
        for i, s in enumerate(splits):
            patients = self.df.loc[self.df['split'] == s].index.values
            d[i] = list(patients)
        path = os.path.join(loc, 'splits.json')
        with open(path, 'w') as file:
            json.dump(d, file, indent=1)

    def getFileName(self, subject, sequence):
        return self.df.at[subject, sequence]
