# Title: preprocessing.py
# Date: 31/01/227
# Author: James Barrett

# Description: Adopts the preprocessing steps defined by Liao et al. (2019) for KDSB17

# Save Pre-Processed Images as NPY to reduce training time
import config
import numpy as np
import nibabel as nib
import scipy.ndimage
from math import prod
from sys import path
import matplotlib.pyplot as plt
from skimage import measure, morphology
path.append('utils/')


def resampling(slices):
    if any(np.asarray(slices.shape) <= config.IMAGE_SIZE):
        dif = config.IMAGE_SIZE - slices.shape
        mod = dif % 2
        dif = dif // 2
        pad = np.maximum(dif, [0, 0, 0])
        pad = tuple(zip(pad, pad + mod))
        slices = np.pad(slices, pad, 'reflect')

    sz = config.IMAGE_SIZE[0]
    if any(np.asarray(slices.shape) >= config.IMAGE_SIZE):
        x, y, z = slices.shape
        x = x // 2 - (sz // 2)
        y = y // 2 - (sz // 2)
        z = z // 2 - (sz // 2)
        image = slices[x:x + sz, y:y + sz, z:z + sz]
        return image


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                               truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                               truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=None, area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if vol_limit is None:
        vol_limit = [0.68, 8.2]
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = {label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], label[-1 - cut_num, 0, 0],
                label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0], label[-1 - cut_num, -1, -1], label[0, 0, mid],
                label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]}
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * prod(spacing) < vol_limit[0] * 1e6 or prop.area * prod(spacing) > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = {label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], label[-1, 0, 0], label[-1, 0, -1],
                label[-1, -1, 0], label[-1, -1, -1]}
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


class LiaoTransform(object):

    def __init__(self):
        """
        :param slices: 3x3x3 ndarray of CT SCAN
        :return: processed scan images as uint8
        """

        # get slices
    def __call__(self, scan_path):
        """
        :param path: str path of scan
        :return: processed image
        """
        # Convert to Houndsfield Unit Scale [Already Handled By Nibabel.Load()]
        self.scan = nib.load(scan_path)
        # Resizing
        self.slices = resampling(np.array(self.scan.get_fdata()))
        # Mask Extraction
        self.mask_extraction()
        # Convex Hull and Dilation
        # Intensity Normalisation
        # crop_scan

        return self.scan


    def mask_extraction(self):
        print("Displaying Results (BW)")
        ''' STEP1 PYTHON Adopted from Liao (2019) '''
        spacing = self.scan.header.get_zooms()
        bw = binarize_per_slice(self.slices, spacing)
        flag = 0
        cut_num = 0
        cut_step = 2
        bw0 = np.copy(bw)
        while flag == 0 and cut_num < bw.shape[0]:
            bw = np.copy(bw0)
            bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5])
            cut_num = cut_num + cut_step

        bw = fill_hole(bw)
        bw1, bw2, bw = two_lung_only(bw, spacing)
        show_slices(self.slices[50:66])
        show_slices(bw[50:66])
        #return case_pixels, bw1, bw2,



        # Remove Top Slices
        # Gaussian Filter (stdv = 1px)
        # Binarized Filter (thresh = -600)
        # Remove 2D connected components smaller than 30 mm2 or having eccentricity greater than 0.99
        # Only 3D Components not touching the matrix corner and having a volume between 0.68 L and 7.5 L are kept.
        # Calculate min distance from image center
        # Select Slices with area > 600mm^2
        # Remove if avg min distance > 62mm
        # Union remaining components for the final mask

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
        # crop scans in all 3 dimensions so that the margin in 10 pixels on every side
        return scan


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(4, 4)
    for i, slice in enumerate(slices):
        axes[i // 4][i % 4].axis("off")
        axes[i// 4][i% 4].imshow(slice.T, cmap="gray", origin="lower")

    plt.show()


