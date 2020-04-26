import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from pathlib import Path

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def get_mean_and_std(dataloader, device=torch.device('cpu')):
    num_samples = 0.
    mean = 0.
    std = 0.
    pbar = tqdm(len(dataloader))
    for batch, _ in dataloader:
        batch = batch.to(device)
        batch_size = batch.size(0)
        batch = batch.view(batch_size, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        num_samples += batch_size
        pbar.update()
    pbar.close()

    mean /= num_samples
    std /= num_samples

    return mean.cpu(), std.cpu()

def save_cm(cm, title, path, normalized=False,
            format="d", xticklabels='auto', yticklabels='auto',
            vmin=None, vmax=None):
    ax = sns.heatmap(
        cm.get(normalized=normalized), annot=True, fmt=format,
        xticklabels=xticklabels, yticklabels=yticklabels or xticklabels,
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f'{title}')
    plt.xlabel('predicted')
    plt.ylabel('ground')
    hm = ax.get_figure()
    hm.savefig(path)
    hm.clf()
