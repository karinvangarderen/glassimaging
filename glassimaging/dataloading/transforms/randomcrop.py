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
        subject = sample['subject']        
        new_h, new_w, new_d = self.output_size
        
        padding = [(0,0),(0,0),(0,0),(0,0)]
        if new_h < h:
            top = np.random.randint(0, h - new_h)
        elif new_h == h:
            top = 0
        else:
            diff = new_h - h
            padding[1] = (int(np.floor(diff/2)), int(np.ceil(diff/2)))
            top = 0
        if new_w < w:
            left = np.random.randint(0, w - new_w)
        elif new_w == w:
            left = 0
        else:
            diff = new_w - w
            padding[2] = (int(np.floor(diff/2)), int(np.ceil(diff/2)))
            left = 0
        if new_d < d:
            front = np.random.randint(0, d - new_d)
        elif new_d == d:
            front = 0
        else:
            diff = new_d - d
            padding[3] = (int(np.floor(diff/2)), int(np.ceil(diff/2)))
            front = 0
       
        image = np.pad(image, padding) 
        image = image[:, top: top + new_h,
                      left: left + new_w, front: front + new_d]

        sample['data'] = image
        if 'seg' in sample:
            segmentation = sample['seg']
            segmentation = np.pad(segmentation, padding[1:])
            segmentation = segmentation[top: top + new_h,
                           left: left + new_w, front: front + new_d]
            sample['seg'] = segmentation
        return sample
