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
from util.dataloader import EMM_LC_Fusion_Loader
from util.preprocessing import LiaoTransform, MRIdataset
from util.nets.Fusion import Fusion
from util.utils import AverageMeter, save_model


def train_model(epoch, model, optim, criterion, train_loader, writer):
    model.train()
    epoch_loss = AverageMeter()
    print_stats = 5  # Every 5 Percent Print Stats

    for batch_idx, data in enumerate(train_loader):
        data = data['scan'].to(device=config.DEVICE).float()
        targets = data['label'].to(device=config.DEVICE).float()
        descriptor = data['descriptor'].to(device=config.DEVICE).float()

        # Forward Pass
        scores = model(data, descriptor)
        loss = criterion(scores, targets.unsqueeze(1))

        # Backward Pass
        optim.zero_grad()
        loss.backward()

        # Adam Step
        optim.step()

        epoch_loss.update(loss.item())

        if batch_idx % print_stats == 0:
            writer.add_scalar('training loss',
                              loss.item(),
                              epoch * len(train_loader) + batch_idx)
            text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(text.format(
                time.strftime("%H:%M:%S"), (batch_idx + 1),
                (len(train_loader)), 100. * (batch_idx + 1) / (len(train_loader)),
                loss.item()))

    print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg


def main(load_path=None, train=True):
    print("Running...")
    # Change CSV for Training
    train_data = EMM_LC_Fusion_Loader(scan_csv='Preprocessed-LIAO-L-Thresh-CSV\\train_file.csv',
                                      desc_csv='Preprocessed-LIAO-L-Thresh-CSV\\train_descriptor.csv',
                                      transform=transforms.Compose([Resize((64, 64, 64))]))
    test_data = EMM_LC_Fusion_Loader(scan_csv='Preprocessed-LIAO-L-Thresh-CSV\\test_file.csv',
                                     desc_csv='Preprocessed-LIAO-L-Thresh-CSV\\test_descriptor.csv',
                                     transform=transforms.Compose([Resize((64, 64, 64))]))
    print(train_data[0])
    print("Loaded Dataset")

    # Over Sampling due to lack of entries with cancer : Limitation -> May Produce Overfitting
    w = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights), replacement=True)

    train_loader = DataLoader(train_data, sampler=w, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = Fusion()
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
        writer = SummaryWriter(config.DATA_DIR + '/models/VGG/logs/runs')
        for epoch in range(epoch, config.NUM_EPOCHS):
            print(f"Epoch {epoch}")

            train_loss = train_model(epoch, model, optimizer, criterion, train_loader, writer)
            print(f"Average Loss in epoch {epoch} :::: {train_loss}")

            test_loss, f1, flag = test(model, test_loader, criterion, writer, epoch=epoch)

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

            # Implement is best
            if is_best:
                save_model(state, model_path=config.DATA_DIR + "/models/VGG/checkpoints/Best.pth")
            else:
                save_model(state, model_path=config.DATA_DIR + "/models/VGG/checkpoints/Last.pth")


def test(model, loader, criterion, writer, epoch=0):
    model.eval()
    epoch_loss = AverageMeter()
    count, correct = 0, 0
    labels, patients, scores, predictions = [], [], [], []

    for batch_idx, data in enumerate(loader):
        data = data['scan'].to(device=config.DEVICE).float()
        target = data['label'].to(device=config.DEVICE).float()

        labels.extend(data['label'].tolist())

        with torch.no_grad():
            out = model(data)

        loss = criterion(out, target.unsqueeze(1))
        epoch_loss.update(loss.item())
        pred = (out > 0).float()
        predictions.extend(pred.tolist())
        scores.extend(out.tolist())
        count += pred.sum()
        correct += (pred * target).sum()

    print(f"Epoch {epoch}")
    print('Val:\n  Loss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, predictions)
    accuracy = sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels)

    print(f"ROC_AUC: {roc} \n     AP: {ap} \n     F1: {f1}\n ACCURACY: {accuracy}")
    print(count)
    print("___________________________")

    writer.add_scalar('ROC_AUC', roc, epoch)
    writer.add_scalar('F1', f1, epoch)

    flag = True
    if count == 0 or count == len(loader.dataset):
        flag = False

    return epoch_loss.avg, f1, flag


if __name__ == "__main__":
    print(config.DEVICE)
    main()
