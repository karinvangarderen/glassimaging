# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:53:06 2019

@author: 043413
"""
import sys
import torch
import numpy as np
import itertools
import logging
from glassimaging.models.utils import createModel


class NetworkVisualizer():

    def __init__(self, model):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.DEBUG)
        self.model = model
        self.device = NetworkVisualizer.getDevice()
        self.model = self.model.to(self.device)

    def setLogger(self, logger):
        self.logger = logger

    @staticmethod
    def loadFromCheckpoint(loc):
        checkpoint = torch.load(loc, map_location=NetworkVisualizer.getDevice())
        net = createModel(checkpoint['model_desc'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return NetworkVisualizer(net)

    @staticmethod
    def getDevice():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def getActivations(self, imagebatch):
        ### Set model in eval mode, so dropout is not used
        self.model.eval()

        ### Reset the model gradients
        self.model.zero_grad()

        ## Send image to device
        imagebatch = imagebatch.to(self.device).float().requires_grad_()

        ### Register the backward hook that saves the gradient on first layer of the model
        ### NB: make sure the model implements the getLastLayer method

        """ Method used to store the gradients """

        def activation_hook(module, inputv, outputv):
            self.activations = outputv[0]

        self.model.getLastLayer().register_forward_hook(activation_hook)
        print("Running model")
        ### Get the output so we know the shape of the tensor
        self.model(imagebatch)
        print("Model finished")

        ### Fetch the activation that was saved by the gradient hook
        acts = self.activations.detach().cpu().numpy()
        del self.activations
        del imagebatch
        self.model.zero_grad()
        acts = acts.astype('float16')
        return acts

    def getNumFeatureMaps(self):
        return self.model.getLastLayer().weight.shape[0]

    def getWholeImageActivations(self, input_array, inputsize, targetsize, weights=None):
        def activation_hook(module, inputv, outputv):
            self.activations = outputv[0]

        self.model.getLastLayer().register_forward_hook(activation_hook)
        channels = self.getNumFeatureMaps()
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
            print("({start}, {end})".format(start=start, end=end))
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
            acts = self.getActivations(inputpatch)
            del inputpatch
            # print(acts.dtype)

            outputselection = np.transpose([start, end])
            slice_obj = (
            slice(None), slice(*outputselection[0]), slice(*outputselection[1]), slice(*outputselection[2]))
            # print(output.shape)
            weighted = np.zeros(acts.shape[1:4])
            if weights is None:
                weights = [1] + [0] * (acts.shape[0] - 1)
            for i in range(0, len(weights)):
                weighted = weighted + weights[i] * acts[i, :, :, :]
            weighted = np.expand_dims(weighted, axis=0)
            print(output.shape)
            print(weighted.shape)
            output[slice_obj] = weighted
            del acts
            # print(output)
            print(sys.getsizeof(output))
            print(sys.getsizeof(self.model))
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        return output

    def getSortedActivationIndices(self, imagebatch, output_index=(10, 10, 10), outputclass=1):
        weights = self.getActivationWeights(imagebatch, output_index, outputclass)
        indices = np.argsort(weights)
        return indices

    def getActivationWeights(self, imagebatch, output_index=(10, 10, 10), outputclass=1):
        """ Method used to store the gradients """

        def gradient_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        ### Set model in eval mode, so dropout is not used
        self.model.eval()

        # imagebatch = torch.from_numpy(imagebatch).float()
        ## Send image to device
        imagebatch = imagebatch.to(self.device).requires_grad_()

        ### Register the backward hook that saves the gradient on first layer of the model
        ### NB: make sure the model implements the getLastLayer method
        self.model.getLastLayer().register_backward_hook(gradient_hook)

        ### Get the output so we know the shape of the tensor
        output = self.model(imagebatch)

        ### Reset the model gradients
        self.model.zero_grad()

        ## Create target with shape of the output tensor
        output_target = output.new_zeros(output.size(), requires_grad=True)
        output_target[0, outputclass, output_index[0], output_index[1], output_index[2]] = 1
        output = torch.sum(output_target * output)

        ### Backprop the gradient from required output to input image
        torch.autograd.grad(output, imagebatch)

        ### Fetch the gradient that was saved by the gradient hook
        grads = self.gradients.data.cpu().numpy()

        weights = np.apply_over_axes(np.sum, grads, [0, 2, 3, 4])[0, :, 0, 0, 0]
        return weights

    def getWeightedActivations(self, imagebatch, output_index=(10, 10, 10), outputclass=1):
        """ Method used to store the gradients """

        def activation_hook(module, inputv, outputv):
            self.activations = outputv[0]

        """ Method used to store the gradients """

        def gradient_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        ### Set model in eval mode, so dropout is not used
        self.model.eval()

        ## Send image to device
        imagebatch = imagebatch.to(self.device).requires_grad_()

        ### Register the backward hook that saves the gradient on first layer of the model
        ### NB: make sure the model implements the getLastLayer method
        self.model.getLastLayer().register_forward_hook(activation_hook)
        self.model.getLastLayer().register_backward_hook(gradient_hook)

        ### Get the output so we know the shape of the tensor
        output = self.model(imagebatch)

        ### Fetch the activation that was saved by the gradient hook
        acts = self.activations.data.numpy()

        ### Reset the model gradients
        self.model.zero_grad()

        ## Create target with shape of the output tensor
        output_target = output.new_zeros(output.size(), requires_grad=True)
        output_target[0, outputclass, output_index[0], output_index[1], output_index[2]] = 1
        output = torch.sum(output_target * output)

        ### Backprop the gradient from required output to input image
        torch.autograd.grad(output, imagebatch)

        ### Fetch the gradient that was saved by the gradient hook
        grads = self.gradients.data.cpu().numpy()

        weights = np.apply_over_axes(np.sum, grads, [0, 2, 3, 4])[0, :, 0, 0, 0]
        indices = np.argsort(weights)
        return acts[indices, :, :, :], weights[indices]

    def getAllActivations(self, imagebatch):
        def activation_hook(module, inputv, outputv):
            self.activations = outputv[0]

        """ Method used to store the gradients """


        ### Set model in eval mode, so dropout is not used
        self.model.eval()

        ## Send image to device
        imagebatch = imagebatch.to(self.device).requires_grad_()

        ### Register the backward hook that saves the gradient on first layer of the model
        ### NB: make sure the model implements the getLastLayer method
        self.model.getLastLayer().register_forward_hook(activation_hook)

        ### Get the output so we know the shape of the tensor
        self.model(imagebatch)

        ### Fetch the activation that was saved by the gradient hook
        acts = self.activations.data.numpy()

        ### Reset the model gradients
        self.model.zero_grad()

        return acts

    def getMultipleLayerRandomActivations(self, imagebatch, num_maps):

        self.activations = []

        def activation_hook(module, inputv, outputv):
            self.activations.append(outputv[:,:num_maps,:,:,:])

        ### Set model in eval mode, so dropout is not used
        self.model.eval()

        ## Send image to device
        imagebatch = imagebatch.to(self.device).requires_grad_()

        ### Register the forward hook that saves the activations
        ### NB: make sure the model implements the getMultipleConvLayers method
        for layer in self.model.getLayersToVisualize():
            layer.register_forward_hook(activation_hook)

        ### Get the output so we know the shape of the tensor
        self.model(imagebatch)

        acts = []

        for act_array in self.activations:
            multiplier = imagebatch.shape[2] / act_array.shape[2]
            print(multiplier)
            if multiplier > 1:
                act_array_new = torch.nn.functional.upsample(act_array, scale_factor=multiplier)
                print(act_array_new.shape)
                acts.append(act_array_new.detach().numpy()[0])

        acts = np.stack(acts)

        ### Reset the model gradients
        self.model.zero_grad()

        return acts


