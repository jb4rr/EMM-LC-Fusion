import torch
import config
from sys import path
path.append('util/')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_model(state, model_path='models/default.pt'):
    torch.save(state, model_path)
    print(f"Saved Epoch in {model_path}")


def plot_confusion_matrix(labels, predicted):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cf_matrix = confusion_matrix(labels, predicted)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(config.DATA_DIR + '/models/Multimodal/EMM-LC-Fusion/no_dae/N1v2/checkpoints/confusion_matrix.png')
    plt.show()