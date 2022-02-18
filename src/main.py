# Title: Main
# Author: James Barrett
# Date: 27/01/22
import config

import torch
import torch.nn as nn
from torch.optim import Adam
from torchio.transforms import Resize
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from sklearn.metrics import f1_score
from utils.dataloader import LUCASDataset
from utils.models import VGG16


def save_model(model, epoch, loss, optim):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, f'./models/model-{epoch}.pt')
    print(f"Saved Epoch as models/model-{epoch}.pt")


def train_model(model, optim, criterion, train_loader, start_epoch=0):
    model.train()
    loss = 0
    #writer = SummaryWriter('./models/logs/runs')
    batch_idx = 1
    data, targets = next(iter(train_loader))
    print(targets)
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        running_loss = 0
        #for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=config.DEVICE).float()
        targets = targets.to(device=config.DEVICE).float()

        # Forward Pass
        scores = model(data)
        pred = torch.round(scores)
        print(f"Predicted: {pred}")
        print(f"Scores   : {scores}")

        loss = criterion(scores, targets.unsqueeze(1))
        # Backward Pass
        optim.zero_grad()
        loss.backward()
        # Adam Step
        optim.step()

        running_loss += loss.item()
        #writer.add_image('Sample Batch', make_grid([i[:,:,32] for i in data]),epoch * len(train_loader) + batch_idx)
        #writer.add_image('Sample Batch', make_grid([i[:, :, 20] for i in data]),
        #                 epoch * len(train_loader) + batch_idx)
        #writer.add_scalar('training loss',
        #                  loss,
        #                  epoch * len(train_loader) + batch_idx)
        print(f"Loss in epoch {epoch} :::: {running_loss / len(train_loader)}")
        # Save Model

        #save_model(model, epoch, loss, optim)


def main(load_path=None, train=True):
    train_data = LUCASDataset('train_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))
    test_data = LUCASDataset('test_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))
    # Over Sampling due to lack of entries with cancer : Limitation -> May Produce Overfitting
    w_sampler = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights), replacement=True)
    train_loader = DataLoader(train_data, sampler=w_sampler, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

    model = VGG16()  # to compile the model
    model = model.to(device=config.DEVICE)  # to send the model for training on either cuda or cpu
    model = model.float()

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)  # Adam seems to be the most popular for deep learning
    epoch = 0

    if train:
        if load_path:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']

        train_model(model, optimizer, criterion, train_loader, start_epoch=epoch)
    else:
        test_model(model, test_loader)


def test_model(model, test_loader):
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device=config.DEVICE)
            targets = targets.to(device=config.DEVICE)

            # Forward Pass
            scores = model(data.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            f1_score_res = f1_score(targets, predictions)
            print(f"Targets: {targets}\nScores: {scores}\nPredictions: {predictions}\nF1 Score: {f1_score_res}")
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )


if __name__ == "__main__":
    main(load_path=None, train=True)
