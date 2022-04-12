from sys import path
from src import config
from torch.utils.data import Dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

path.append('util/')


class EMM_LC_Fusion_Loader_Lucas(Dataset):
    def __init__(self, scan_csv=None, desc_csv=None):
        self.root_dir = config.DATA_DIR
        self.image_size = np.asarray((128,128,128))

        self.task = -2  # -2=cancer; -1=nodule/mass

        labels = pd.read_csv(os.path.join(self.root_dir, scan_csv))
        self.labels = labels.set_index('patient_id').T.to_dict('list')
        descriptor = pd.read_csv(os.path.join(self.root_dir, desc_csv))
        self.descriptor = descriptor.set_index('patient_id').T.to_dict('list')
        self.idx = list(self.labels.keys())
        self.weights = self.weights_balanced()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        patient = self.idx[idx]
        label = self.labels[patient][self.task]
        descriptor = self.descriptor[patient]
        descriptor = descriptor[1:12] + descriptor[13:-1]
        descriptor = np.array(descriptor, dtype=np.float32)
        if os.path.exists(self.root_dir + '/Data/Preprocessed-LUCAS/' + str(patient) + '.npy'):
            image = np.load(self.root_dir + '/Data/Preprocessed-LUCAS/' + str(patient) + '.npy')
        else:
            image = load_image('SCANS/' + str(patient) + '.nii.gz', self.root_dir)

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
            np.save(self.root_dir + '/Data/Preprocessed-LUCAS/' + str(patient) + '.npy', image)

        return {'scan': np.expand_dims(image, 0), 'descriptor': descriptor, 'label': label, 'id': patient}

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


class EMM_LC_Fusion_Loader_Liao(Dataset):
    def __init__(self, scan_csv=None, desc_csv=None, transform=None):
        # Set Root Data Directory
        self.root = config.DATA_DIR
        self.desc_labels = None
        self.scan_labels = None
        self.scan = None
        self.task = -2  # Used to Index Column -2 = Cancer
        self.transform = transform

        if desc_csv is not None:
            # Format and Remove Redundant Columns From DataFrame
            self.desc_labels = pd.read_csv(os.path.join(self.root, desc_csv))
            self.desc_labels = self.desc_labels.drop(['Benign_cons', 'Malignant_gra', 'x<3mm_mass'], axis=1)
            self.desc_labels = self.desc_labels.set_index('patient_id').T.to_dict('list')

            # Replace with value 1 if greater than 1 (Lack of Clarity In Dataset as to what these values mean)
            for key, value in self.desc_labels.items():
                self.desc_labels[key] = [1 if ele > 1 else ele for ele in self.desc_labels[key]]

            self.idx = list(self.desc_labels.keys())

        if scan_csv is not None:
            self.scan_labels = pd.read_csv(os.path.join(self.root, scan_csv))
            self.scan_labels = self.scan_labels.set_index('patient_id').T.to_dict('list')
            self.idx = list(self.scan_labels.keys())
            self.weights = self.weights_balanced()

    def __len__(self):
        # Return Length of Labels Regardless of what Loader is Loading
        labels = next(l for l in [self.scan_labels, self.desc_labels] if l)
        return len(labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.idx[idx]
        if self.desc_labels is not None:
            desc_labels = self.desc_labels[patient][1:-1]
            desc_labels = torch.unsqueeze(torch.tensor(desc_labels), 0)
        else:
            desc_labels = torch.empty(1)

        if self.scan_labels is not None:
            scan_labels = self.scan_labels[patient][self.task]
            scan_labels = float(scan_labels)

            preprocessed_img_dir = os.path.join(self.root, "Preprocessed-LIAO-L-Thresh", str(patient) + ".npy")
            scan = np.load(preprocessed_img_dir)
            scan = torch.unsqueeze(torch.tensor(scan), 0)
            if self.transform:
                scan = self.transform(scan)
        else:
            scan = torch.empty(1)
            scan_labels = torch.empty(1)

        return {'scan': scan, 'label': scan_labels, 'descriptor': desc_labels, 'id': patient}

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.scan_labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.scan_labels[val][self.task]]
        return weight

def load_image(patient, root_dir):
    im = nib.load(os.path.join(root_dir, patient))
    image = im.get_fdata()
    return image


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
    dataset = EMM_LC_Fusion_Loader(scan_csv='Data\\Preprocessed-LUCAS-CSV\\train_file.csv',
                                      desc_csv='Data\\Preprocessed-LUCAS-CSV\\train_descriptor.csv')
    print(len(dataset[0]['descriptor']))
    show_slices([dataset[0]['scan'][0, 64, :, :]],total_cols=1)


