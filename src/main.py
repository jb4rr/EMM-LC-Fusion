# Title: Main
# Author: James Barrett
# Date: 27/01/22

import torch
import config
from torch.utils.data import DataLoader, sampler
from utils.dataloader import LUCASDataset

def main():
    train_data = LUCASDataset('train_file.csv')
    test_data = LUCASDataset('test_file.csv')
    w_sampler = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights))
    print(w_sampler)
    train_loader = DataLoader(train_data, sampler=w_sampler,batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

    for i_batch, sample_batched in enumerate(train_loader):
        images_batch, labels_batch = sample_batched['image'], sample_batched['label']
        print(labels_batch)



if __name__ == "__main__":
    main()
