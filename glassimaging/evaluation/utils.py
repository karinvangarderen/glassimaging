# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import nilearn.plotting as plotting
import os
import logging
import pandas as pd
from glassimaging.training.standardTrainer import StandardTrainer
import itertools
from glassimaging.dataloading.brats18 import Brats18


def segmentNifti(images, segfiles, trainer, inputsize, targetsize, savepaths):
    input_array = np.stack(images)
    output = segmentWholeArray(input_array, inputsize, targetsize, trainer)

    for i in range(0, len(savepaths)):
        ######## Load segmentation to get affine and header information
        nifti = nib.load(segfiles[i])
        hdr = nifti.header.copy()
        ff = nifti.affine.copy()
        segmentation_img = nib.Nifti1Image(output[i], ff, hdr)
        nib.save(segmentation_img, savepaths[i])
    return output


def segmentNiftiWithSeg(images, trainer, inputsize, targetsize, savepaths, nifti_header_source):
    input_array = np.stack(images)
    output = segmentWholeArray(input_array, inputsize, targetsize, trainer)

    for i in range(0, len(savepaths)):
        ######## Load segmentation to get affine and header information
        nifti = nib.load(nifti_header_source[i])
        hdr = nifti.header.copy()
        ff = nifti.affine.copy()
        segmentation_img = nib.Nifti1Image(output[i], ff, hdr)
        nib.save(segmentation_img, savepaths[i])
    return output


def segmentWholeArray(input_array, inputsize, targetsize, trainer):
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
        outputpatch = trainer.inferWithImage(inputpatch)
        outputpatch = outputpatch.cpu().numpy()
        outputpatch = np.argmax(outputpatch, axis=1)

        outputselection = np.transpose([start, end])
        slice_obj = (slice(None), slice(*outputselection[0]), slice(*outputselection[1]), slice(*outputselection[2]))

        output[slice_obj] = outputpatch
    return output

def plotResultImage(datamanager, resultpath, savepath, subject, output_type='save'):
    basefile = datamanager.getFileName(subject, 't1')
    segfile = datamanager.getFileName(subject, 'seg')
    if output_type == 'save':
        figure, (axes1, axes2) = plt.subplots(2, 1, figsize=(10, 10))
        cut = plotting.find_xyz_cut_coords(segfile, activation_threshold=0.5)
        plotting.plot_roi(resultpath, basefile, cut_coords=cut, axes=axes1, title='Result')
        plotting.plot_roi(segfile, basefile, cut_coords=cut, axes=axes2, title='Target')
        plt.savefig(os.path.join(savepath, 'result_' + subject + '.png'))
        plt.close()
        logging.info('Subject ' + str(subject) + ' plotted with image ' + basefile + '.')
    elif output_type == 'show':
        figure, (axes1, axes2) = plt.subplots(2, 1, figsize=(10, 10))
        plotting.plot_roi(savepath, basefile, axes=axes1, title='Result')
        plotting.plot_roi(segfile, basefile, axes=axes2, title='Target')
        plt.show()
        plt.close()


def plotResultImageWithoutDatamanager(path_seg, path_t1, path_result, savepath, subject, output_type='save'):
    if output_type == 'save':
        figure, (axes1, axes2) = plt.subplots(2, 1, figsize=(10, 10))
        cut = plotting.find_xyz_cut_coords(path_seg, activation_threshold=0.5)
        plotting.plot_roi(path_result, path_t1, cut_coords=cut, axes=axes1, title='Result')
        plotting.plot_roi(path_seg, path_t1, cut_coords=cut, axes=axes2, title='Target')
        plt.savefig(os.path.join(savepath, 'result_' + subject + '.png'))
        plt.close()
        logging.info('Subject ' + str(subject) + ' plotted with image ' + path_t1 + '.')
    elif output_type == 'show':
        figure, (axes1, axes2) = plt.subplots(2, 1, figsize=(10, 10))
        plotting.plot_roi(path_result, path_t1, axes=axes1, title='Result')
        plotting.plot_roi(path_seg, path_t1, axes=axes2, title='Target')
        plt.show()
        plt.close()


def plotResultImageWithoutGT(path_t1, path_result, savepath, subject, output_type='save'):
    if output_type == 'save':
        plotting.plot_roi(path_result, path_t1, title=str(subject))
        plt.savefig(os.path.join(savepath, 'result_' + subject + '.png'))
        plt.close()
        logging.info('Subject ' + str(subject) + ' plotted with image ' + path_t1 + '.')
    elif output_type == 'show':
        plotting.plot_roi(path_result, path_t1, title=str(subject))
        plt.show()
        plt.close()


def getPerformanceMeasures(result, truth):
    TT = np.sum(np.logical_and(result, truth))
    FP = np.sum(np.logical_and(result, np.logical_not(truth)))
    FN = np.sum(np.logical_and(np.logical_not(result), truth))
    TN = np.sum(np.logical_and(np.logical_not(result), truth))
    if (TT + FP + FN) > 0:
        dice = (2 * TT) / (2 * TT + FP + FN)
    else:
        dice = 1
    return (dice, TT, FP, FN, TN)


def evaluatePerformanceNifti(datamanager, patchsize=(36, 36, 36), tmploc='./tmp', splits=(4), savepath='./log',
                             output_type='show', only_first=False, uid=-1, sequences=None):
    modelpath = os.path.join(tmploc, str(uid) + '_model.pickle')
    savepath = os.path.join(savepath, str(uid))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    net = torch.load(modelpath)
    logging.info('Model loaded from ' + modelpath + '.')
    trainer = StandardTrainer(net)
    niftiloader = datamanager.getNiftiLoader(splits=splits, sequences=sequences, batch_size=2)
    all_dice = []
    results = pd.DataFrame(columns=['sample', 'subject', 'class', 'TP', 'FP', 'FN', 'TN', 'dice'])
    for i_batch, sample_batched in enumerate(niftiloader):
        (images, segs, headers, affines, subjects) = sample_batched
        resultpaths = [os.path.join(savepath, s + '_segmented.nii.gz') for s in subjects]
        classifications = segmentNifti(images, affines, headers, trainer, patchsize, resultpaths)
        for i in range(0, len(subjects)):
            seg = segs[i]
            plotResultImage(datamanager, resultpaths[i], savepath, subjects[i], output_type=output_type)
            for c in range(0, 5):
                truth = seg == c
                positive = classifications[i] == c
                (dice, TT, FP, FN, TN) = getPerformanceMeasures(positive, truth)
                results = results.append(
                    {'sample': i_batch, 'class': c, 'subject': subjects[i], 'TP': TT, 'FP': FP, 'FN': FN, 'TN': TN,
                     'dice': dice}, ignore_index=True)
            class_whole = classifications[i] > 0
            truth_whole = seg > 0
            (dice, TT, FP, FN, TN) = getPerformanceMeasures(class_whole, truth_whole)
            all_dice.append(dice)
            logging.info('Nifti image segmented for ' + subjects[i] + '. Dice: ' + str(dice))
            results = results.append(
                {'sample': i_batch, 'class': 'whole', 'subject': subjects[i], 'TP': TT, 'FP': FP, 'FN': FN, 'TN': TN,
                 'dice': dice}, ignore_index=True)
        if only_first: break
    dice_mean = sum(all_dice) / len(all_dice)
    plt.boxplot(all_dice)
    plt.savefig(os.path.join(savepath, 'boxplot_dice_btd.png'))
    plt.close()
    results.to_csv(os.path.join(savepath, 'results_eval.csv'))
    logging.info('evaluation finished. Dice coefficient: ' + str(dice_mean))

def createNewDatamanager(location, dataset, nsplits):
    if dataset == 'Brats':
        return Brats18.fromFile(location, nsplits=nsplits)

def logDataLoader(dataloader, directory):
    loader = iter(dataloader)
    batch = next(loader)
    img = batch["data"]
    seg = batch["seg"]
    subj = batch["subject"]
    print(img.shape)
    for i in range(0, img.shape[0]):
        plt.figure(figsize=(10, 10))
        sl = img.shape[3] // 2
        for j in range(0, img.shape[1]):
            plt.subplot(3, 2, j + 1)
            plt.imshow(img[i, j, :, sl], cmap="gray")  # only grayscale image here
            plt.colorbar()
        sl = seg.shape[2] // 2
        plt.subplot(3, 2, 5)
        plt.imshow(seg[i, :, sl], cmap="gray")  # only grayscale image here
        plt.colorbar()
        plt.savefig(os.path.join(directory, 'batch_{}_input.png'.format(subj[i])))
        plt.close()


def plotResultImageWithoutGT(path_t1, path_result, savepath, subject, output_type='save'):
    if output_type == 'save':
        plotting.plot_roi(path_result, path_t1, title=str(subject))
        plt.savefig(os.path.join(savepath, 'result_' + subject + '.png'))
        plt.close()
        logging.info('Subject ' + str(subject) + ' plotted with image ' + path_t1 + '.')
    elif output_type == 'show':
        plotting.plot_roi(path_result, path_t1, title=str(subject))
        plt.show()
        plt.close()
