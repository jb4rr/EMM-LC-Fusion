# Model Summary
#   Returns Vector of [1,1, no_features]
#   Calculates the labels for each subject
#   Rich Features to Be Extracted from this model

import config
import time
import itertools

from torch import nn, optim, cuda, no_grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from util.dataloader import EMM_LC_Fusion_Loader
from util.nets.DAE import DenoisingAutoEncoder
from util.utils import AverageMeter, save_model


def main():
    # Define Dataset
    train_data = EMM_LC_Fusion_Loader(desc_csv='Preprocessed-LIAO-L-Thresh-CSV/train_descriptor.csv')
    test_data = EMM_LC_Fusion_Loader(desc_csv='Preprocessed-LIAO-L-Thresh-CSV/test_descriptor.csv')

    # Define DataLoader
    train_loader = DataLoader(train_data, batch_size=8,
                              num_workers=config.NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8,
                             num_workers=config.NUM_WORKERS, shuffle=True)

    # Define Model
    model = DenoisingAutoEncoder()
    model = model.to(device=config.DEVICE)
    model = model.float()

    # Define Loss Function with L1 Regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Train Model
    writer = SummaryWriter(config.DATA_DIR+f"/models/Unimodal/DAE/{config.DAE_NUM}/logs")
    cuda.empty_cache()
    train(model, criterion, optimizer, train_loader, test_loader, writer)

    # Test Model
    cuda.empty_cache()
    test(model, criterion, test_loader, writer, epoch=0)


def train(model, criterion, optimizer, loader, test_loader, writer):
    # Set Model to update gradients

    # Define Logging Parameters
    print_stats = 5
    epoch_loss = AverageMeter()
    best_f1 = 0
    # Add Tensorboard Logging

    for epoch in range(0, config.NUM_EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(loader):

            # Get Data
            data = sample['descriptor'].to(device=config.DEVICE).float()

            # Forward Pass : DAE returns both encoded and decoded score. Ignore Encoded Here
            _, scores = model(data)
            loss = criterion(scores, data)

            # Add L1 Regularization Term to Prevent Creating an Identity Function
            l1_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss = loss + l1_lambda * l1_norm

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()

            # Adam Step
            optimizer.step()

            # Update Average Loss Counter
            epoch_loss.update(loss.item())

            if batch_idx % print_stats == 0:
                writer.add_scalar('Training Loss', loss.item(), epoch * len(loader) + batch_idx)
                text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(text.format(time.strftime("%H:%M:%S"), (batch_idx + 1), (len(loader)),
                                  100. * (batch_idx + 1) / (len(loader)), loss.item()))

        # End Of Epoch :
        print(f"Loss in epoch {epoch} :::: {epoch_loss.avg / len(loader)}")

        # Test Model
        test_loss, f1 = test(model, criterion, test_loader, writer, epoch=epoch)


        # Save Model
        is_best = best_f1 <= f1
        best_f1 = max(best_f1, f1)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': [epoch_loss.avg, test_loss],
            'lr': config.LR,
            'f1': f1,
            'best_f1': best_f1}

        if is_best:
            save_model(state, model_path=config.DATA_DIR+f"/models/Unimodal/DAE/{config.DAE_NUM}/checkpoints/BEST_DAE.pth")
        else:
            save_model(state, model_path=config.DATA_DIR+f"/models/Unimodal/DAE/{config.DAE_NUM}/checkpoints/LAST_DAE.pth")
        epoch_loss.reset()


def test(model, criterion, loader, writer, epoch=0):
    model.eval()
    # Define Logging Parameters
    epoch_loss = AverageMeter()
    labels, scores, predictions = [], [], []

    for batch_idx, sample in enumerate(loader):

        # Get Data
        data = sample['descriptor'].to(device=config.DEVICE).float()
        labels.extend(data.tolist())

        with no_grad():
            _, out = model(data)

        loss = criterion(out, data)
        predictions.extend((out >= 0.5).float().tolist())
        epoch_loss.update(loss.item())
    labels = list(itertools.chain(*list(itertools.chain(*labels))))
    predictions = list(itertools.chain(*list(itertools.chain(*predictions))))

    f1 = f1_score(labels, predictions)
    print("\n------ VALIDATION ------")
    print('      Loss: {:.6f}'.format(epoch_loss.avg))
    print('   F-Score: {:.6f}\n'.format(f1))
    writer.add_scalar('Validation Loss', epoch_loss.avg, epoch)
    writer.add_scalar('F1 Score', f1, epoch)
    return epoch_loss.avg, f1


def visualise_data():
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(r'E:\University of Gloucestershire\Year 4\Dissertation\train_descriptor.csv')
    #sn.heatmap(df.corr()[['Cancer']])
    #plt.savefig('./correlation_matrix_all.png')
    # Remove Columns with all Zeros
    df = df.drop(['Benign_cons', 'Malignant_gra','x<3mm_mass', 'Index', 'patient_id'], axis=1)
    df1 = df.iloc[:, 37:].copy()
    #df1['Cancer'] = df['Cancer']
    sn.set(font_scale=0.5)
    sn.heatmap(df1.corr()[['Cancer']])
    #plt.savefig('./correlation_matrix_less.png')
    plt.show()

if __name__ == "__main__":
    main()
