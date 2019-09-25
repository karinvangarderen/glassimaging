import SimpleITK as sitk
import numpy as np
import argparse
import os

def main(path, resultpath, path_mask=None):

    #Load image
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    filter = sitk.N4BiasFieldCorrectionImageFilter()
    filter.SetNumberOfThreads(1)
    if path_mask is not None:
        mask = sitk.ReadImage(path_mask, sitk.sitkUInt8)
        output = filter.Execute(img, mask)
    else:
        output = filter.Execute(img)
    sitk.WriteImage(output, resultpath)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resample an image to 1x1x1 voxels")
    parser.add_argument('--img', type=str, required=True, help="Image path")
    parser.add_argument('--brainmask', type=str, required=False, help="Mask path")
    parser.add_argument('--imgout', type=str, required=True, help="Image output path")
    args = parser.parse_args()
    main(args.img, args.imgout, path_mask=args.brainmask)
