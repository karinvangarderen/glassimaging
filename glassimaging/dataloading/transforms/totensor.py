# -*- coding: utf-8 -*-
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, segmentation = sample['data'], sample['seg']

        sample['data'] = torch.from_numpy(image)
        sample['seg'] = torch.from_numpy(segmentation)
        return sample
