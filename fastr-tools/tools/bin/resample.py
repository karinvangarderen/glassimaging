import SimpleITK as sitk
import numpy as np
import argparse
import os

def main(path, resultpath, mask=None, maskout=None):

    #Load image
    img = sitk.ReadImage(path)

    #resample image and optionally mask
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    spacing = [1.0, 1.0, 1.0]
    orig_size = np.array(img.GetSize(), dtype=np.int)
    orig_spacing = img.GetSpacing()
    new_size = [int(np.ceil(a[0] * (a[1]/a[2]))) for a in zip(orig_size, orig_spacing, spacing)]
    resample.SetSize(new_size)
    resample.SetOutputSpacing(spacing)

    resample.SetNumberOfThreads(1)

    resampled_img = resample.Execute(img)
    sitk.WriteImage(resampled_img, resultpath)

    if mask is not None:
        mask_image = sitk.ReadImage(mask)
        resampled_mask = resample.Execute(mask_image)
        sitk.WriteImage(resampled_mask, maskout)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resample an image to 1x1x1 voxels")
    parser.add_argument('--img', type=str, required=True, help="Image path")
    parser.add_argument('--mask', type=str, required=False, help="Mask path")
    parser.add_argument('--imgout', type=str, required=True, help="Image output path")
    parser.add_argument('--maskout', type=str, required=False, help="Mask output path")
    args = parser.parse_args()
    main(args.img, args.imgout, mask=args.mask, maskout=args.maskout)
