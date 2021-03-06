# Title: Main
# Author: James Barrett
# Date: 27/01/22
import matplotlib.pyplot as plt

import config
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import Adam, lr_scheduler
from torchio.transforms import Resize, RandomFlip, RandomAffine
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from util.dataloader import EMM_LC_Fusion_Loader

from util.nets.AlignedXception import AlignedXception
from util.utils import AverageMeter, save_model, get_lr


def main(load_path=None, train=True):
    print("Running...")

    #np.random.seed(12345)
    #torch.manual_seed(12345)
    #torch.cuda.manual_seed_all(12345)

    #cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Change CSV for Training
    train_data = EMM_LC_Fusion_Loader(scan_csv='Preprocessed-LIAO-L-Thresh-CSV/train_file.csv',
                                      desc_csv='Preprocessed-LIAO-L-Thresh-CSV/train_descriptor.csv',
                                      transform=transforms.Compose([RandomFlip(2, flip_probability=0.5),
                                                                    RandomAffine(degrees=(-20, 20, 0, 0, 0, 0),
                                                                                 default_pad_value=170),
                                                                    Resize((128, 128, 128))]))
    test_data = EMM_LC_Fusion_Loader(scan_csv='Preprocessed-LIAO-L-Thresh-CSV/test_file.csv',
                                     desc_csv='Preprocessed-LIAO-L-Thresh-CSV/test_descriptor.csv',
                                     transform=transforms.Compose([Resize((128, 128, 128))]))

    print("Loaded Dataset")

    # Over Sampling due to lack of entries with cancer : Limitation -> May Produce Over fitting
    w = sampler.WeightedRandomSampler(train_data.weights, len(train_data.weights), replacement=True)

    train_loader = DataLoader(train_data, sampler=w, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = AlignedXception(BatchNorm=nn.BatchNorm3d, filters=[16, 32, 64, 128, 128, 256])
    model = model.to(device=config.DEVICE)
    model = model.float()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    annealing = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
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

            model.load_state_dict(checkpoint['model_state_dict'])

        writer = SummaryWriter(config.DATA_DIR + '/models/Unimodal/ALX/logs/runs')
        for epoch in range(epoch, config.NUM_EPOCHS):
            lr = get_lr(optimizer)
            print(f"Epoch {epoch} with LR == {lr}")

            train_loss = train_model(epoch, model, optimizer, criterion, train_loader, writer)
            print(f"Average Loss in epoch {epoch} :::: {train_loss}")

            test_loss, f1, flag = test(model, test_loader, criterion, writer, epoch=epoch)

            # Reduce on Plateau
            annealing.step(test_loss)

            is_best = False
            if flag:
                # Only Improve Best Model after 10 epochs
                if epoch > 10:
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
                # Only Save after 10 epochs
                if epoch > 10:
                    save_model(state, model_path=config.DATA_DIR + "/models/Unimodal/ALX/checkpoints/Best.pth")
            else:
                save_model(state, model_path=config.DATA_DIR + "/models/Unimodal/ALX/checkpoints/Last.pth")

            if lr <= (config.LR / (10 ** 4)):
                print('Stopping training: learning rate is too small')
                break


def train_model(epoch, model, optim, criterion, train_loader, writer):
    model.train()
    epoch_loss = AverageMeter()
    print_stats = 5  # Every 5 Percent Print Stats

    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, sample in enumerate(train_loader):
        data = sample['scan'].to(device=config.DEVICE).float()
        targets = sample['label'].to(device=config.DEVICE).float()
        targets = torch.stack([1 - targets, targets], dim=1)
        optim.zero_grad()
        # Forward Pass
        with torch.cuda.amp.autocast():
            scores, _ = model(data)
            loss = criterion(scores, targets)

        # Backward Pass
        scaler.scale(loss).backward()  # loss.backward()

        # Adam Step
        scaler.step(optim)
        scaler.update()        # optim.step()

        epoch_loss.update(loss.item())

        if batch_idx % print_stats == 0:
            writer.add_scalar('training loss',
                               epoch_loss.avg,
                              epoch * len(train_loader) + batch_idx)
            text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(text.format(
                time.strftime("%H:%M:%S"), (batch_idx + 1),
                (len(train_loader)), 100. * (batch_idx + 1) / (len(train_loader)),
                loss.item()))

    print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg


def test(model, loader, criterion, writer, epoch=0):
    model.eval()
    epoch_loss = AverageMeter()
    count, correct = 0, 0
    labels, patients, scores, predictions = [], [], [], []

    for batch_idx, sample in enumerate(loader):
        data = sample['scan'].to(device=config.DEVICE).float()
        target = sample['label'].to(device=config.DEVICE).float()
        labels.extend(sample['label'].tolist())

        with torch.no_grad():
            out, _ = model(data)
        loss = criterion(out, torch.stack([1 - target, target], dim=1))
        epoch_loss.update(loss.item())

        confidence = F.softmax(out, dim=1)
        scores.extend(confidence[:, 1].tolist())

        pred = torch.argmax(confidence, dim=1)
        predictions.extend(pred.tolist())
        count += pred.sum()
        correct += (pred * target).sum()

    print(f"Epoch {epoch}")
    print('Val:\n     Loss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    roc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, predictions)
    accuracy = sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels)

    print(f"  ROC_AUC:   {roc} \n     AP: {ap} \n       F1: {f1}\n ACCURACY: {accuracy}")
    print("___________________________")

    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(config.DATA_DIR+f'/models/Unimodal/ALX/checkpoints/AUC-ROC Curve/Epoch-{epoch}-Curve.png')
    plt.clf()

    writer.add_scalar('ROC_AUC', roc, epoch)
    writer.add_scalar('F1', f1, epoch)
    writer.add_scalar('Average Precision', ap, epoch)
    writer.add_scalar('Validation Loss', epoch_loss.avg, epoch)

    flag = True
    if count == 0 or count == len(loader.dataset):
        flag = False

    return epoch_loss.avg, f1, flag


if __name__ == "__main__":
    print(config.DEVICE)
    main()
