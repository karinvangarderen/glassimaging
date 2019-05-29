# -*- coding: utf-8 -*-

import torch
import logging
import sys
import numpy as np
import itertools
import nibabel as nib
from glassimaging.models.utils import createModel


class StandardEvaluator:

    def __init__(self, net):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.DEBUG)
        self.net = net
        self.device = StandardEvaluator.getDevice()
        if torch.cuda.device_count() == 2:
            self.logger.info('Two devices being used.')
            self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1]).cuda()
            self.parallel = True
        else:
            self.parallel = False
            self.logger.info('Number of devices being used: {n}'.format(n=torch.cuda.device_count()))
            self.net = self.net.to(self.device)

    def setLogger(self, logger):
        self.logger = logger

    @staticmethod
    def getDevice():
       return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def initFromDesc(desc):
        net = createModel(desc)
        return StandardEvaluator(net)

    @staticmethod
    def loadFromCheckpoint(loc):
        checkpoint = torch.load(loc, map_location=StandardEvaluator.getDevice())
        net = createModel(checkpoint['model_desc'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return StandardEvaluator(net)

    def saveModel(self, loc):
        if self.parallel:
            model_desc = self.net.module.getDesc()
            state_dict = self.net.module.state_dict()
        else:
            model_desc = self.net.getDesc()
            state_dict = self.net.state_dict()
        torch.save({
            'model_desc': model_desc,
            'model_state_dict': state_dict
            }, loc)

    def getModelDesc(self):
        if self.parallel:
            return self.net.module.getDesc()
        else:
            return self.net.getDesc()

    def inferWithImage(self, image):
        self.net = self.net.eval()
        with torch.no_grad():
            torchimage = image.float()
            torchimage = torchimage.to(self.device)
            output = self.net(torchimage)
            output = output.detach()
        return output

    def segmentWholeArray(self, input_array, inputsize, targetsize):
        outputsize = np.array(input_array.shape[2:5])
        inputsize = np.array(inputsize)
        targetsize = np.array(targetsize)
        crop = (inputsize - targetsize) // 2
        output = np.zeros(input_array.shape[0:1] + input_array.shape[2:5])
        topleft = [np.arange(0, outputsize[i] + 1, targetsize[i]) for i in range(0, 3)]
        bottomright = [np.arange(targetsize[i], outputsize[i] + 1 + targetsize[i], targetsize[i]) for i in range(0, 3)]

        startlocations = itertools.product(*topleft)
        endlocations = itertools.product(*bottomright)
        for start, end in zip(startlocations, endlocations):
            start = np.array(start)
            end = np.array(end)
            end = np.amin([end, outputsize], axis=0)
            start = np.amin([start, outputsize - targetsize], axis=0)
            zeros = np.zeros(3, dtype=int)
            startsel = np.amax([start - crop, zeros], axis=0)
            endsel = np.amin([end + crop, outputsize], axis=0)
            padding_start = np.amax([crop - start, zeros], axis=0)
            padding_end = np.amax([crop + end - outputsize, zeros], axis=0)

            selection = np.transpose([startsel, endsel])
            slice_obj = (slice(None), slice(None), slice(*selection[0]), slice(*selection[1]), slice(*selection[2]))
            selected_image = input_array[slice_obj]
            padding = np.transpose([padding_start, padding_end])
            padding = [(0, 0), (0, 0)] + [tuple(padding[i]) for i in range(0, 3)]
            inputpatch = np.pad(selected_image, padding, 'constant')
            inputpatch = torch.from_numpy(inputpatch)
            outputpatch = self.inferWithImage(inputpatch)
            outputpatch = outputpatch.cpu().numpy()
            outputpatch = np.argmax(outputpatch, axis=1)
            outputselection = np.transpose([start, end])
            slice_obj = (
            slice(None), slice(*outputselection[0]), slice(*outputselection[1]), slice(*outputselection[2]))

            output[slice_obj] = outputpatch
        return output

    def segmentVolumeWithOverlap(self, input_array, patchsize, n_overlap=4, n_out=5):
        stepsize = [s / n_overlap for s in patchsize]
        padding = ((0,0), (0,0), (int(stepsize[0] * (n_overlap -1)), int(stepsize[0] * (n_overlap -1))),
                   (int(stepsize[1] * (n_overlap -1)), int(stepsize[1] * (n_overlap -1))),
                   (int(stepsize[2] * (n_overlap -1)), int(stepsize[2] * (n_overlap -1))))
        input = np.pad(input_array, padding, mode='reflect')

        output = np.zeros(input_array.shape[0:1] + (n_out,) + input.shape[2:5])
        inputsize = input.shape[2:5]
        topleft = [np.arange(0, inputsize[i] - patchsize[i] + 1, stepsize[i]) for i in range(0, 3)]
        bottomright = [np.arange(patchsize[i], inputsize[i] + 1, stepsize[i]) for i in range(0, 3)]

        startlocations = itertools.product(*topleft)
        endlocations = itertools.product(*bottomright)
        for start, end in zip(startlocations, endlocations):
            start = np.array(start, dtype=int)
            end = np.array(end, dtype=int)
            selection = np.transpose([start, end])
            slice_obj = (slice(None), slice(None), slice(*selection[0]), slice(*selection[1]), slice(*selection[2]))
            selected_image = input[slice_obj]
            inputpatch = torch.from_numpy(selected_image)
            outputpatch = self.inferWithImage(inputpatch)
            outputpatch = outputpatch.cpu().numpy()
            output[slice_obj] += outputpatch / n_overlap

        mean_output = np.argmax(output, axis=1)
        mean_output = mean_output[:, padding[2][0]:-padding[2][1], padding[3][0]:-padding[3][1], padding[4][0]:-padding[4][1]]

        return mean_output

    def segmentNifti(self, images, segfiles, targetsize, savepaths):
        input_array = np.stack(images)

        output = self.segmentWholeArray(input_array, targetsize, targetsize)

        for i in range(0, len(savepaths)):
            ######## Load segmentation to get affine and header information
            nifti = nib.load(segfiles[i])
            hdr = nifti.header.copy()
            ff = nifti.affine.copy()
            segmentation_img = nib.Nifti1Image(output[i], ff, hdr)
            nib.save(segmentation_img, savepaths[i])
        return output

