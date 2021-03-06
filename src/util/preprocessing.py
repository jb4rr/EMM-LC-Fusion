# Title: preprocessing.py
# Date: 31/01/227
# Author: James Barrett

# Description: Adopts the preprocessing steps defined by Liao et al. (2019) for KDSB17

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import scipy.ndimage
import warnings
from PIL import Image
from nilearn.image import resample_img
from tqdm import tqdm
from pathlib import Path as Plib
from src import config
from sys import path
from skimage import measure
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from scipy.ndimage.interpolation import zoom
import random
import math
import logging
sample_factor = 0.5
path.append('util/')


def binarize_per_slice(image, spacing, intensity_th=-500, sigma=1, area_th=5, eccen_th=.999, bg_patch_size=1):
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


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=3e3, dist_th=62):
    # in some cases, several top layers need to be removed first
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
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
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


def mask_extraction(scan, slices):
    ''' STEP1 PYTHON Adopted from Liao (2019) '''
    # Remove Top Slices
    # Gaussian Filter (stdv = 1px)
    # Binarized Filter (thresh = -600)
    # Remove 2D connected components smaller than 30 mm2 or having eccentricity greater than 0.99
    # Only 3D Components not touching the matrix corner and having a volume between 0.68 L and 7.5 L are kept.
    # Calculate min distance from image center
    # Select Slices with area > 6000mm^2
    # Remove if avg min distance > 62mm
    # Union remaining components for the final mask
    spacing = scan.header.get_zooms()
    spacing = np.array(spacing, dtype=np.float32)
    spacing = np.roll(spacing, 1)
    bw = binarize_per_slice(slices, spacing)

    flag = 0
    cut_num = 2
    cut_step = 2
    bw0 = np.copy(bw)

    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0, 7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return bw1, bw2, spacing


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

class LiaoTransform(object):

    def __init__(self):
        """
        :param slices: 3x3x3 ndarray of CT SCAN
        :return: processed scan images as uint8
        """

        # get slices


    def __call__(self, scan, save=''):
        """
        :param path: str path of scan
        :return: processed image
        """
        exampled_preprocessing = []
        resolution = np.array([1, 1, 1])
        # -------------------------------------------------------------------------------------------------------#
        self.scan = scan
        # -------------------------------------------------------------------------------------------------------#
        #                                         RESIZE OF SCAN Factor=0.25
        logging.debug(f"Scan Size: {self.scan.get_fdata().shape}") # \n   Header: {self.scan.header.get_zooms()}")
            self.scan = resample_img(self.scan, target_affine=self.scan.affine/0.5, interpolation='nearest')
        logging.debug(f"Resized to {self.scan.get_fdata().shape}") # \n   Header: {self.scan.header.get_zooms()}")
        # -------------------------------------------------------------------------------------------------------#
        #                                           GET SLICES
        self.slices = self.scan.get_fdata()
        self.slices = np.stack([(self.slices[:, :, s]) for s in range(self.slices.shape[-1])]).astype(np.int16)
        #show_slices([self.slices[i] for i in range(0,self.slices.shape[0],5)])
        exampled_preprocessing.append(self.scan.get_fdata()[:, :, int(sample_factor * self.slices.shape[0])])
        # -------------------------------------------------------------------------------------------------------#
        #                                           CREATE MASK
        logging.debug("Extracting Mask")
        m1, m2, spacing = mask_extraction(self.scan, self.slices)
        Mask = m1 + m2
        logging.debug("Mask Extracted")
        exampled_preprocessing.append(Mask[int(sample_factor * self.slices.shape[0]), :, :])
        # show_slices(exampled_preprocessing, total_cols=2, titles=['Original', 'Mask'])
        # -------------------------------------------------------------------------------------------------------#
        xx, yy, zz = np.where(Mask)
        # Error if mask is none.
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])

        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        self.slices[np.isnan(self.slices)] = -2000
        sliceim = lumTrans(self.slices)
        logging.debug("Applying Mask")
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        exampled_preprocessing.append(sliceim[int(sample_factor * self.slices.shape[0]), :, :])
        bones = sliceim * extramask > bone_thresh
        logging.debug("Dilating Mask")
        sliceim[bones] = pad_value
        self.slices = sliceim
        logging.debug("Completed Preprocessing")
        #show_slices(exampled_preprocessing, total_cols=3)

        # --------------------------------------------If SAVE = True -----------------------------------------------#
        if save:
            logging.debug(f"Saving Image in {save}")
            self.save_as_numpy(save)
        return self.slices

    def save_as_numpy(self, im_path):
        np.save(im_path, self.slices)


def show_slices(slices, total_cols=6, titles=['Original', 'Mask', 'Output'], save_fig=False):
    """
        Function to display row of image slices
        ref: https://towardsdatascience.com/dynamic-subplot-layout-in-seaborn-e777500c7386
        author: Daniel Deutsch
    """
    num_plots = len(slices)
    total_rows = num_plots // total_cols + 1
    _, axs = plt.subplots(total_rows, total_cols)
    axs = axs.flatten()
    axs = axs.flatten()
    if len(titles) == len(slices):
        for img, ax, title in zip(slices, axs, titles):
            ax.set_title(title, size=5)
            ax.axis("off")
            ax.imshow(img, cmap="gray")
        if save_fig:
            plt.savefig(f'../../images/1_lung.png')
        plt.show()
    else:
        for img, ax in zip(slices, axs):
            ax.axis("off")
            ax.imshow(img, cmap="gray")
        if save_fig:
            plt.savefig(f'../../images/1_lung.png')
        plt.show()


class MRIdataset(object):
    def __init__(self):
        pass

    def __call__(self, scan, save ='', image_size=(256,256,256)):
        self.scan = scan.get_fdata()

        if any(np.asarray(self.scan.shape) <= image_size):
            dif = image_size - self.scan.shape
            mod = dif % 2
            dif = dif // 2
            pad = np.maximum(dif, [0, 0, 0])
            pad = tuple(zip(pad, pad + mod))
            self.scan = np.pad(self.scan, pad, 'reflect')

        sz = image_size[0]
        if any(np.asarray(self.scan.shape) >= image_size):
            x, y, z = self.scan.shape
            x = x // 2 - (sz // 2)
            y = y // 2 - (sz // 2)
            z = z // 2 - (sz // 2)
            self.scan = self.scan[x:x + sz, y:y + sz, z:z + sz]
        # Stats obtained from the MSD dataset
        self.scan = np.clip(self.scan, a_min=-1024, a_max=326)
        self.scan = (self.scan - 159.14433291523548) / 323.0573880113456
        self.scan = np.expand_dims(self.scan, 0)

        if save:
            self.save_as_numpy(save)
        return self.scan

    def save_as_numpy(self, save_path):
        print(f"Saved as {name} in {save_path}")
        np.save(save_path, self.scan)


def get_sample(saved_dir, num_samples=16, sample_point=0.5):
    # Get Random Sample of process data
    flst = []
    # Get List of Files
    files = Plib(saved_dir).glob('*')
    for f in files:
        flst.append(f)
    # get random index's
    files_index = random.sample(range(0, len(flst)), num_samples)

    img_list = []
    name_list = []
    for i in files_index:
        # Load Numpy Data
        scan = np.load(str(flst[i]))
        img_list.append(scan[0, int(scan.shape[1] * sample_point), :, :])
        # Get name for each Numpy Image
        name_list.append(str(flst[i]).split('\\')[-1].split('.')[0])
    # Show Slices
    show_slices(img_list, total_cols=int(math.sqrt(num_samples)), titles=name_list)

if __name__ == "__main__":
    # Set Logging Level:
    #logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

    # Set RAW data file
    scans_dir = str(os.path.join(config.DATA_DIR, "SCANS"))

    # Set directory to save to
    saved_dir = str(os.path.join(config.DATA_DIR, "Preprocessed-LIAO-L-Thresh"))

    # Get Sample
    #get_sample(r"D:\University of Gloucestershire\Year 4\Dissertation\Preprocessed-LIAO-OLD", num_samples=25)

    # Initialise Class Method
    #preprocessor = LiaoTransform()

    # Load 1 Previous Scan
    #scan = np.load(r"D:\University of Gloucestershire\Year 4\Dissertation\Preprocessed-LIAO\4047389.npy")
    #plt.imshow(scan[int(sample_factor * scan.shape[0]), :, :], cmap="gray")
    #plt.show()

    # See If it Improves
    #preprocessor(nib.load(r"D:\University of Gloucestershire\Year 4\Dissertation\SCANS\4047389.nii.gz"))

    # Get File List
    files = Plib(scans_dir).glob('*')

    # Iterate through files
    #for file in files:

        # Get File Name
    #    name = str(file).split("\\")[-1].split('.')[0] + ".npy"
        # Create Save Directory
    #    saved_path = str(os.path.join(saved_dir, name))
    #    if not os.path.exists(saved_path) and name not in open("./L-Thresh-Failed.txt").read():
    #        logging.debug(f"Processing {file}")
    #        try:
                # Process File
    #            preprocessor(nib.load(file), save=saved_path)
    #        except:
                # If Error, skip pre-processing and append file name to failed.txt
    #            logging.warning(f"FAILED: Skipping {file}")
    #            f = open("./L-Thresh-Failed.txt", "a")
    #            f.write('\n')
    #            f.write(name)
    #            f.close()
    #    else:
            # Supports pausing and restarting of pre-processing
    #        logging.debug(f"{file} already processed")

    #get_sample(saved_dir, num_samples=25)
