# Model Summary
#   Returns Vector of [1,1, no_features]
#   Calculates the labels for each subject
#   Rich Features to Be Extracted from this model

import config
import time

import torch
from torch import nn, optim, cuda, no_grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from util.dataloader import DAE
from util.models import DenoisingAutoEncoder
from util.utils import AverageMeter


def main():
    # Define Dataset
    train_data = DAE('train_descriptor.csv', transform=False)
    test_data = DAE('test_descriptor.csv', transform=False)
    # Define DataLoader
    train_loader = DataLoader(train_data, batch_size=32,
                              num_workers=config.NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32,
                             num_workers=config.NUM_WORKERS, shuffle=True)

    # Define Model
    model = DenoisingAutoEncoder()
    model = model.to(device=config.DEVICE)
    model = model.float()

    # Define Loss Function with L1 Regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Train Model
    cuda.empty_cache()
    train(model, criterion, optimizer, train_loader, test_loader)

    # Test Model
    cuda.empty_cache()
    test(model, criterion, optimizer, test_loader)


def train(model, criterion, optimizer, loader, test_loader):
    # Set Model to update gradients


    # Define Logging Parameters
    print_stats = 2
    epoch_loss = AverageMeter()
    batch_loss = AverageMeter()

    # Add Tensorboard Logging
    writer = SummaryWriter('./models/DAE/logs/runs')

    for epoch in range(0, 100):
        model.train()
        for batch_idx, data in enumerate(loader):

            # Get Data
            data = data.to(device=config.DEVICE).float()

            # Forward Pass
            scores = model(data)
            loss = criterion(scores, data)

            # Add L1 Regularization Term to Prevent Creating an Identity Function
            l1_lambda = 0.001
            l1_norm = sum(p.abs().sum()
                          for p in model.parameters())

            loss = loss + l1_lambda * l1_norm

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()

            # Adam Step
            optimizer.step()

            # Update Average Loss Counters
            batch_loss.update(loss.item())
            epoch_loss.update(loss.item())

            if batch_loss.count % print_stats == 0:
                writer.add_scalar('training loss', loss.item(), epoch * len(loader) + batch_idx)
                text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(text.format(time.strftime("%H:%M:%S"), (batch_idx + 1), (len(loader)),
                                  100. * (batch_idx + 1) / (len(loader)), batch_loss.avg))

        # End Of Epoch :
        batch_loss.reset()
        print(f"Loss in epoch {epoch} :::: {epoch_loss.avg / len(loader)}")

        # Test Model?
        test(model, criterion, test_loader)


def test(model, criterion, loader):
    model.eval()

    # Define Logging Parameters
    print_stats = 5
    epoch_loss = AverageMeter()
    labels, scores, predictions = [], [], []
    count, correct = 0, 0

    for batch_idx, data in enumerate(loader):

        # Get Data
        data = data.to(device=config.DEVICE).float()
        labels.append(data.tolist()[0])

        with no_grad():
            out = model(data)

        loss = criterion(out, data)
        epoch_loss.update(loss.item())
        prediction = torch.round(out).float()
        prediction[prediction <= 0.0] = 0
        predictions.append(prediction.tolist()[0])
        scores.extend(out.tolist())
        count += prediction.sum()
        correct += (prediction * data).sum()

    print('--- Val: \tLoss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    accuracy = sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels)

    print(f"ACCURACY: {accuracy}")
    print(count)
    print("___________________________")






if __name__ == "__main__":
    main()
