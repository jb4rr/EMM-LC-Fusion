from sys import path
from src import config
from skimage import color
from torch.utils.data import Dataset
from nibabel import processing

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

path.append('utils/')


class LUCASDataset(Dataset):
    '''
        Description: Custom DataLoader for LUCAS Dataset
    '''

    def __init__(self, csv_file, transform=None):
        self.root = config.DATA_DIR
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
        image_dir = os.path.join(self.root, "SCANS", str(patient) + ".nii.gz")
        if self.transform:
            image = self.transform(image_dir)

        return np.expand_dims(image, 0), float(label)

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(16, 16)

    for i, slice in enumerate(slices):
        axes[i // 16][i % 16].axis("off")
        axes[i// 16][i% 16].imshow(slice.T, cmap="gray", origin="lower")

    plt.show()


if __name__ == '__main__':
    dataset = LUCASDataset('train_file.csv')
    epi_img_data = dataset[0]['image']
    """ Function to display row of image slices """
    show_slices([epi_img_data[:, :, :, a_slice] for a_slice in range(epi_img_data.shape[-1])])



