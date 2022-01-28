from sys import path
path.append('utils/')

import config
import nibabel as nib
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class LUCASDataset(Dataset):
    '''
        Description: Custom DataLoader for LUCAS Dataset
    '''
    def __init__(self, csv_file, transform=None):
        self.root = config.DATA_DIR
        labels = pd.read_csv(os.path.join(self.root, csv_file))
        self.labels = labels.set_index('patient_id').T.to_dict('list')
        self.idx = self.idx = list(self.labels.keys())
        self.task = -2 # Used to Index Column -2 = Cancer
        self.image_size = np.asarray((256,256,256))
        # Omitting removing samples with smaller size
        self.weights = self.weights_balanced()
        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.idx[idx]
        print(self.labels)
        label = self.labels[patient][self.task]
        image_dir = os.path.join(self.root, "SCANS", str(patient) + ".nii.gz")
        image = np.array(nib.load(image_dir).dataobj)

        #if any(np.asarray(image.shape) <= self.image_size):
        #    dif = self.image_size - image.shape
        #    mod = dif % 2
        #    dif = dif // 2
        #    pad = np.maximum(dif, [0, 0, 0])
        #    pad = tuple(zip(pad, pad + mod))
        #    image = np.pad(image, pad, 'reflect')

        #sz = self.image_size[0]
        #if any(np.asarray(image.shape) >= self.image_size):
        #    x, y, z = image.shape
        #    x = x // 2 - (sz // 2)
        #    y = y // 2 - (sz // 2)
        #    z = z // 2 - (sz // 2)
        #    image = image[x:x + sz, y:y + sz, z:z + sz]
        # Stats obtained from the MSD dataset
        image = np.clip(image, a_min=-1024, a_max=326)
        image = (image - 159.14433291523548) / 323.0573880113456

        return {'image': np.expand_dims(image, 0), 'label': label}

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight

    def transform(self):
        pass


def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

if __name__ == '__main__':
    dataset = LUCASDataset('train_file.csv')
    epi_img_data = dataset[0]['image']
    import matplotlib.pyplot as plt
    print(epi_img_data.shape)
    """ Function to display row of image slices """
    slice_0 = epi_img_data[:,26, :, :]
    slice_1 = epi_img_data[:,:, 30, :]
    slice_2 = epi_img_data[:,:, :, 200]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")
    plt.show()