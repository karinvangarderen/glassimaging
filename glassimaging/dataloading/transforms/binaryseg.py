# -*- coding: utf-8 -*-
import torch


class BinarySegmentation(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        seg = sample['seg']
        seg = seg > 0.5
        seg = seg.astype('int')
        sample['seg'] = seg
        return sample
