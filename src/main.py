# Title: Main
# Author: James Barrett
# Date: 27/01/22
import config
import torch
import torch.nn as nn
import time
import os
import csv
from torch.optim import Adam
from torchio.transforms import Resize
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from util.dataloader import LUCASDataset
from util.preprocessing import LiaoTransform, MRIdataset
from util.models import VGG16
from util.utils import AverageMeter


def save_model(model, epoch, loss, optim):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, f'./models/model-.pt')
    print(f"Saved Epoch as models/model-{epoch}.pt")


def train_model(epoch, model, optim, criterion, train_loader):
    model.train()
    epoch_loss = AverageMeter()
    batch_loss = AverageMeter()
    print_stats = len(train_loader) // 5
    writer = SummaryWriter('./models/logs/runs')
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=config.DEVICE).float()
        targets = targets.to(device=config.DEVICE).float()

        # Forward Pass
        scores = model(data)
        loss = criterion(scores, targets.unsqueeze(1))

        # Backward Pass
        optim.zero_grad()
        loss.backward()

        # Adam Step
        optim.step()

        batch_loss.update(loss.item())
        epoch_loss.update(loss.item())

        if batch_loss.count % print_stats == 0:
            writer.add_scalar('training loss',
                              batch_loss.avg,
                              epoch * len(train_loader) + batch_idx)
            text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(text.format(
                time.strftime("%H:%M:%S"), (batch_idx + 1),
                (len(train_loader)), 100. * (batch_idx + 1) / (len(train_loader)),
                batch_loss.avg))
            batch_loss.reset()
    print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg


def main(load_path=None, train=True):

    train_data = LUCASDataset('train_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))
    test_data = LUCASDataset('test_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))

    #train_data = MRIdataset('train_file.csv', 'train_descriptor.csv', config.DATA_DIR, config.IMAGE_SIZE, transforms=transforms.Compose([Resize((64, 64, 64))]))
    #test_data = MRIdataset('test_file.csv', 'test_descriptor.csv', config.DATA_DIR, config.IMAGE_SIZE, transforms=transforms.Compose([Resize((64, 64, 64))]))

    # Over Sampling due to lack of entries with cancer : Limitation -> May Produce Overfitting
    w_sampler = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights), replacement=True)

    train_loader = DataLoader(train_data, sampler=w_sampler, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS)

    model = VGG16()  # to compile the model
    model = model.to(device=config.DEVICE)  # to send the model for training on either cuda or cpu
    model = model.float()


    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config.LR)  # Adam seems to be the most popular for deep learning
    epoch = 0
    best_f1 = 0

    if train:
        torch.cuda.empty_cache()
        if load_path:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
        for epoch in range(epoch, config.NUM_EPOCHS):
            train_loss = train_model(epoch, model, optimizer, criterion, train_loader)
            print(f"Loss in epoch {epoch} :::: {train_loss / len(train_loader)}")
            # Save Model
            test_loss, f1, flag = test(model, test_loader, '../models/logs/', criterion, training=True)

            is_best = False
            if flag:
                is_best = best_f1 < f1
                best_f1 = max(best_f1, f1)

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': [train_loss, test_loss],
                'lr': config.LR,
                'f1': f1,
                'best_f1': best_f1}

            # Implemenent is best
            if is_best:
                save_model(model, epoch, test_loss, optimizer, path="src/models/Best.pth")
    else:
        pass
        # Implement Model Predict


def test(model, loader, save_path, criterion, training=True):
    model.eval()
    model.float()
    epoch_loss = AverageMeter()
    count, correct = 0, 0
    labels, patients, scores, predictions = [], [], [], []

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=config.DEVICE).float()
        target = targets.to(device=config.DEVICE).float()
        labels.extend(targets.tolist())

        with torch.no_grad():
            out = model(data)

        loss = criterion(out, target.unsqueeze(1))
        epoch_loss.update(loss.item())

        confidence = torch.softmax(out, dim=1)
        scores.extend(confidence.tolist())

        pred = torch.argmax(confidence, dim=1)
        predictions.extend(pred.tolist())
        count += pred.sum()
        correct += (pred * target).sum()
    print('--- Val: \tLoss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, predictions)
    print(f"ROC_AUC: {roc} \nAP: {ap} \n F1: {f1}")

    if not training:
        print('ROC', roc, 'AP', ap, 'F1', f1)
        rows = zip(patients, scores)
        with open(os.path.join(save_path, 'confidence.csv'), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['ROC:', roc])
            writer.writerow(['AP:', ap])
            writer.writerow(['F1:', f1])
            for row in rows:
                writer.writerow(row)

    count = count.sum()
    flag = True
    if count == 0 or count == len(loader.dataset):
        flag = False

    return epoch_loss.avg, f1, flag


if __name__ == "__main__":
    # Clear GPU CACHE

    main(load_path=None, train=True)
