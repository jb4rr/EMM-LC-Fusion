

# Model Summary
#   Returns Vector of [1,1,76]
#   Calculates the labels for each subject
#   Rich Features to Be Extracted from this model

import config

from torch import nn, optim, cuda
from torch.utils.data import DataLoader

from util.dataloader import DAE
from util.models import DenoisingAutoEncoder



def main():
    # Define Dataset
    train_data = DAE('train_descriptor.csv', transform=False)
    test_data = DAE('test_descriptor.csv', transform=False)

    # Define Dataloader
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS)

    # Define Model
    model = DenoisingAutoEncoder()  # to compile the model
    model = model.to(device=config.DEVICE)  # to send the model for training on either cuda or cpu
    model = model.float()

    # Define Loss Function with L1 Regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Train Model
    cuda.empty_cache()
    train(model, criterion, optimizer, train_loader)

    # Test Model
    cuda.empty_cache()
    test(model, criterion, optimizer, test_loader)


def train(model, criterion, optimizer, loader):
    model.train()
    for epoch in range(0, 50):
        for batch_idx, data in enumerate(loader):
            # Get Data

            data = data.to(device=config.DEVICE).float()

            # Forward Pass
            scores = model(data)
            loss = criterion(scores, data)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()

            # Adam Step
            optimizer.step()

            if batch_loss.count % print_stats == 0:
                writer.add_scalar('training loss',
                                  batch_loss.avg,
                                  epoch * len(train_loader) + batch_idx)
                text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(text.format(
                    time.strftime("%H:%M:%S"), (batch_idx + 1),
                    (len(train_loader)), 100. * (batch_idx + 1) / (len(train_loader)),
                    batch_loss.avg))


def test(model, criterion, optimizer, loader):
    model.eval()


if __name__ == "__main__":
    main()