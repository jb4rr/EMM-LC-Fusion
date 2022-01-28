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
        self.task = -2
        self.image_size = np.asarray((256,256,256))
        # Omitting removing samples with smaller size
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.idx[idx]

        label = self.labels[patient][self.task]
        image_dir = os.path.join(self.root, "SCANS", str(patient) + ".nii.gz")
        image = a = np.array(nib.load(image_dir).dataobj)

        if any(np.asarray(image.shape) <= self.image_size):
            dif = self.image_size - image.shape
            mod = dif % 2
            dif = dif // 2
            pad = np.maximum(dif, [0, 0, 0])
            pad = tuple(zip(pad, pad + mod))
            image = np.pad(image, pad, 'reflect')

        sz = self.image_size[0]
        if any(np.asarray(image.shape) >= self.image_size):
            x, y, z = image.shape
            x = x // 2 - (sz // 2)
            y = y // 2 - (sz // 2)
            z = z // 2 - (sz // 2)
            image = image[x:x + sz, y:y + sz, z:z + sz]
        # Stats obtained from the MSD dataset
        image = np.clip(image, a_min=-1024, a_max=326)
        image = (image - 159.14433291523548) / 323.0573880113456

        return {'image': np.expand_dims(image, 0), 'label': label}


    def transform(self):
        pass


if __name__ == '__main__':
    dataset = LUCASDataset('train_file.csv')
    print(dataset[0]['image'].shape)