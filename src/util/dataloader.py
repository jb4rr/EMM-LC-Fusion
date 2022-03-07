from sys import path
from src import config
from skimage import color
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from nibabel import processing
import os
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

path.append('util/')


class DAE(Dataset):
    def __init__(self, csv_file, transform=None):
        self.root = config.DATA_DIR
        self.labels = pd.read_csv(os.path.join(self.root, csv_file))

        self.labels = self.labels.set_index('patient_id').T.to_dict('list')

        # Replace with value 1 if greater than 1 (Lack of Clarity In Dataset as to what these values mean)

        for key, value in self.labels.items():
            self.labels[key] = [1 if ele > 1 else ele for ele in self.labels[key]]

        self.idx = self.idx = list(self.labels.keys())
        self.factors = config.FACT_IDX

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get Patient
        patient = self.idx[idx]

        # Exclude Cancer Diagnosis From Training
        label = [list(self.labels[patient])[i] for i in self.factors]
        label = torch.unsqueeze(torch.tensor(label), 0)
        return label


class VGG16_Loader(Dataset):
    '''
        Description: Custom DataLoader for LUCAS Dataset
    '''

    def __init__(self, csv_file, transform=None, preprocessed=False):
        self.root = config.DATA_DIR
        self.preprocessed = preprocessed
        labels = pd.read_csv(os.path.join(self.root, csv_file))
        self.labels = labels.set_index('patient_id').T.to_dict('list')
        self.idx = self.idx = list(self.labels.keys())
        self.task = -2  # Used to Index Column -2 = Cancer
        self.image_size = np.asarray((64, 64, 64))
        # Omitting removing samples with smaller size may cause error
        self.weights = self.weights_balanced()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.idx[idx]
        label = self.labels[patient][self.task]

        if self.preprocessed == True:
            image_dir = os.path.join(self.root, "Preprocessed-LIAO", str(patient) + ".npy")
            scan = np.load(image_dir)
        else:
            image_dir = os.path.join(self.root, "SCANS", str(patient) + ".nii.gz")
            scan = nib.load(image_dir)

        if self.transform:
            scan = torch.unsqueeze(torch.tensor(scan), 0)
            scan = self.transform(scan)
        # show_slices(scan[0,20:56,:,:], total_cols=6)

        return scan, float(label)

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N / float(count[i])
        print(weight_per_class)
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight


def show_slices(slices, total_cols=6):
    """
        Function to display row of image slices
        ref: https://towardsdatascience.com/dynamic-subplot-layout-in-seaborn-e777500c7386
        author: Daniel Deutsch
    """
    num_plots = len(slices)
    total_rows = num_plots // total_cols + 1
    _, axs = plt.subplots(total_rows, total_cols)
    axs = axs.flatten()
    for img, ax in zip(slices, axs):
        ax.axis("off")
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
    plt.show()


if __name__ == '__main__':
    dataset = VGG16_loader('train_file.csv')
    epi_img_data = dataset[0]['image']
    """ Function to display row of image slices """
    show_slices([epi_img_data[:, :, :, a_slice] for a_slice in range(epi_img_data.shape[-1])])
