# -*- coding: utf-8 -*-
import numpy as np


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['data']

        h, w, d = image.shape[1:]
        new_h, new_w, new_d = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        front = np.random.randint(0, d - new_d)

        image = image[:, top: top + new_h,
                      left: left + new_w, front: front + new_d]

        sample['data'] = image
        if 'seg' in sample:
            segmentation = sample['seg']
            segmentation = segmentation[top: top + new_h,
                           left: left + new_w, front: front + new_d]
            sample['seg'] = segmentation
        return sample
