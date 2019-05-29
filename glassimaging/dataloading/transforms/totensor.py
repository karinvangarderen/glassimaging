# -*- coding: utf-8 -*-
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['data']
        sample['data'] = torch.from_numpy(image)
        if 'seg' in sample:
            segmentation = sample['seg']
            sample['seg'] = torch.from_numpy(segmentation)
        return sample
