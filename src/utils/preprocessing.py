# Title: preprocessing.py
# Date: 31/01/227
# Author: James Barrett

# Description: Adopts the preprocessing steps defined by Liao et al. (2019) for KDSB17


class LiaoTransform(object):

    def __init__(self):
        """
        :param slices: 3x3x3 ndarray of CT SCAN
        :return: processed scan images as uint8
        """

        # get slices
    def __call__(self, scan):
        self.slices = [scan[:, :, a_slice] for a_slice in range(scan.shape[-1])]
        print(self.slices[0].shape)

        # Convert to Houndsfield Unit Scale [Already Handled By Nibabel.Load()]
        # Mask Extraction
        # Convex Hull and Dilation
        # Intensity Normalisation
        # crop_scan
        return scan


    def mask_extraction(self, scan):
        # Remove Top Slices
        # Gaussian Filter (stdv = 1px)
        # Binarized Filter (thresh = -600)
        # Remove 2D connected components smaller than 30 mm2 or having eccentricity greater than 0.99
        # Only 3D Components not touching the matrix corner and having a volume between 0.68 L and 7.5 L are kept.
        # Calculate min distance from image center
        # Select Slices with area > 600mm^2
        # Remove if avg min distance > 62mm
        # Union remaining components for the final mask
        return scan

    def convex_hull_dilation(self, scan):
        # Divide the scan into 2 parts (left / right)
        # Iteratively erode each side to the same volume
        # Dilate both components back to original size
        # Intersection with raw mask is now for two lungs seperately
        # Replace each 2d slice with convex hull
        # Dilate by further 10 voxels
        # if convex hull of 2d slice is > 1.5 times the original mask is kept
        return scan

    def intensity_normalisation(self, scan):
        # transform from HU to uint8
        # clip data from [-1200, 600]
        # linearly transform to [0,255]
        # Apply Mask (Multiply)
        # Everything outside mask fill with 170
        # All Values greater than 210 replaced with 170
        # fill bones with 170
        return scan

    def crop_scan(self, scan):
        # crop scans in all 3 dimesions so that the margin in 10 pixels on every side
        return scan