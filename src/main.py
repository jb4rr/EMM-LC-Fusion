# Title: Main
# Author: James Barrett
# Date: 27/01/22
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torchio.transforms import Resample, Resize
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from utils.dataloader import LUCASDataset
from utils.models import VGG16


def main():
    train_data = LUCASDataset('train_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64,64,64))]))
    test_data = LUCASDataset('test_file.csv')
    w_sampler = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights))
    train_loader = DataLoader(train_data, sampler=w_sampler, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)


    model = VGG16()  # to compile the model
    model = model.to(device=config.DEVICE)  # to send the model for training on either cuda or cpu
    model = model.float()
    ## Loss and optimizer
    learning_rate = 1e-4  # I picked this because it seems to be the most used by experts
    load_model = True
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam seems to be the most popular for deep learning

    for epoch in range(50):  # I decided to train the model for 50 epochs
        loss_ep = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=config.DEVICE)
            targets = targets.to(device=config.DEVICE)
            print(f"Image Type {type(data)} \nImage Shape {data.shape}" )
            print(f"Image Type {type(targets)} \nImage Shape {targets.unsqueeze(1).shape}" )
            ## Forward Pass
            optimizer.zero_grad()
            scores = model(data.float())
            loss = criterion(scores, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            print(f"Loss in epoch {epoch} :::: {loss_ep / len(train_loader)}")

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device=config.DEVICE)
                targets = targets.to(device=config.DEVICE)
                ## Forward Pass
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )

if __name__ == "__main__":
    main()
