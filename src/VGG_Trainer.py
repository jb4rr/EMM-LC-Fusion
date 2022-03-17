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
from util.dataloader import VGG16_Loader
from util.preprocessing import LiaoTransform, MRIdataset
from util.models import VGG16
from util.utils import AverageMeter, save_model


def train_model(epoch, model, optim, criterion, train_loader, writer):
    model.train()
    epoch_loss = AverageMeter()
    batch_loss = AverageMeter()
    print_stats = 5  # Every 5 Percent Print Stats

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
    print("Running...")
    # Change CSV for Training
    train_data = VGG16_Loader('Processed-LIAO-CSV/train_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))
    test_data = VGG16_Loader('Processed-LIAO-CSV/test_file.csv', preprocessed=True, transform=transforms.Compose([Resize((64, 64, 64))]))
    print("Loaded Dataset")

    # Over Sampling due to lack of entries with cancer : Limitation -> May Produce Overfitting
    w = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights), replacement=True)

    train_loader = DataLoader(train_data, sampler=w, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = VGG16()
    model = model.to(device=config.DEVICE)
    model = model.float()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.LR)

    epoch = 0
    best_f1 = 0

    print("Training")
    if train:
        torch.cuda.empty_cache()
        if load_path:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_f1 = checkpoint['f1']
        writer = SummaryWriter('./models/VGG/logs/runs')
        for epoch in range(epoch, config.NUM_EPOCHS):
            print(f"Epoch {epoch}")

            train_loss = train_model(epoch, model, optimizer, criterion, train_loader, writer)
            print(f"Loss in epoch {epoch} :::: {train_loss}")

            test_loss, f1, flag = test(model, test_loader, '../models/logs/', criterion, writer,training=True)

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
                save_model(state, model_path="src/models/vgg/checkpoints/Best.pth")
            else:
                save_model(state, model_path="./models/DAE/checkpoints/N20/LAST_DAE.pth")


def test(model, loader, save_path, criterion, writer,training=True, epoch=0):
    model.eval()
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
        pred = (out > 0).float()
        predictions.extend(pred.tolist())
        scores.extend(out.tolist())
        count += pred.sum()
        correct += (pred * target).sum()

    print('--- Val: \tLoss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, predictions)
    accuracy = sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels)

    print(f" ROC_AUC: {roc} \n     AP: {ap} \n     F1: {f1}\n ACCURACY: {accuracy}")
    print(count)
    print("___________________________")

    writer.add_scalar('training loss', epoch_loss.avg, epoch)
    writer.add_scalar('ROC_AUC', roc, epoch)
    writer.add_scalar('F1', f1, epoch)

    flag = True
    if count == 0 or count == len(loader.dataset):
        flag = False

    return epoch_loss.avg, f1, flag

if __name__ == "__main__":
    print(config.DEVICE)
    main()