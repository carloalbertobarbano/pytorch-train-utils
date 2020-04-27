import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tqdm.notebook as tqdmn
from tqdm import tqdm

def in_ipynb():
    try:
        cfg = get_ipython().config
        return True
    except Exception as e:
        return False

if in_ipynb():
    tqdm = tqdmn.tqdm

from . import utils

def summarize_metrics(metrics):
    summarizable = dict(filter(lambda m: m[1].summarizable if hasattr(m[1], 'summarizable') else True, metrics.items()))
    return ' - '.join(list(map(lambda kv: '{}: {:.4f}'.format(kv[0], float(str(kv[1]))), summarizable.items())))

def report_metrics(metrics, end='\n'):
    print(summarize_metrics(metrics), end=end, flush=True)

def run(model, dataloader, criterion, optimizer, metrics, phase, device=torch.device('cuda:0'), weight=None):
    num_batches = 0.
    loss = 0.

    if phase == 'train':
        model.train()
    else:
        model.eval()

    for metric in metrics:
        metric.reset()


    for data, labels in tqdm(dataloader, desc=phase, leave=False):
        data, labels = data.to(device), labels.to(device)

        running_loss = 0.
        output = None

        with torch.set_grad_enabled(phase == 'train'):
            output = model(data)
            running_loss = criterion(output, labels, weight=weight)

        if phase == 'train':
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.zero_grad()
            else:
                for opt in optimizer:
                    opt.zero_grad()

            running_loss.backward()

            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.step()
            else:
                for opt in optimizer:
                    opt.step()

        for metric in metrics:
            metric.accumulate(output, labels)

        loss += running_loss.item()
        num_batches += 1

    logs = { metric.__name__: copy.deepcopy(metric) for metric in metrics }
    logs.update({'loss': loss / num_batches})
    return logs

def make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params=None):
    checkpoint = {
        'epoch': epoch, 'model': model.state_dict(), 'metrics': metrics
    }

    if isinstance(optimizer, torch.optim.Optimizer):
        checkpoint.update({'optimizer': optimizer.state_dict()})
    else:
        checkpoint.update({'optimizers': []})
        for opt in optimizer:
            checkpoint['optimizers'].append(opt.state_dict())

    if checkpoint_params is not None:
        checkpoint.update(checkpoint_params)
    return checkpoint

def is_better(a, b, mode='min'):
    if mode == 'min':
        return a < b
    elif mode == 'max':
        return a > b

    return False

def plot_losses(train, val, test, name, path):
    df = pd.DataFrame({'train': train, 'val': val, 'test': test})
    ax = sns.lineplot(data=df)
    ax.set_title(name)
    hm = ax.get_figure()
    hm.savefig(path)
    hm.clf()

def fit(model, train_dataloader, val_dataloader, test_dataloader, test_every,
        criterion, optimizer, scheduler, metrics, n_epochs, name, path='',
        weight={'train': None, 'val': None, 'test': None},
        metric_choice='loss', mode='min', device=torch.device('cuda:0'), checkpoint_params=None, callbacks=None):
    utils.ensure_dir(name)

    best_metric = 0.
    best_model = None

    train_losses = []
    val_losses = []
    test_losses = []

    test_logs = {'loss': 1.}

    for epoch in range(n_epochs):

        train_logs = run(
            model=model, dataloader=train_dataloader,
            criterion=criterion, weight=weight['train'], optimizer=optimizer,
            metrics=metrics, phase='train', device=device
        )

        if 'train' in callbacks and callbacks['train'] is not None:
            callbacks['train']()

        val_logs = run(
            model=model, dataloader=val_dataloader,
            criterion=criterion, weight=weight['val'], optimizer=None,
            metrics=metrics, phase='val', device=device
        )

        if 'val' in callbacks and callbacks['val'] is not None:
            callbacks['val']()

        print(f'Epoch: {epoch:03d} | VAL ', end='')
        report_metrics(val_logs, end=' | TRAIN ')
        report_metrics(train_logs, end=' |\n')

        if scheduler is not None:
            scheduler.step(val_logs['loss'])

        torch.save(
            make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params),
            os.path.join(path, f'{name}/final.pt')
        )

        if best_model is None or is_better(float(str(val_logs[metric_choice])), best_metric, mode):
            best_metric = float(str(val_logs[metric_choice]))
            best_model = copy.deepcopy(model)
            torch.save(
                make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params),
                os.path.join(f'{name}/best.pt')
            )

        if test_dataloader is not None and (epoch+1) % test_every == 0:
            test_logs = test(
                model=model, test_dataloader=test_dataloader,
                criterion=criterion, metrics=metrics,
                device=device,
                weight=weight['test']
            )

            if 'test' in callbacks and callbacks['test'] is not None:
                callbacks['test']()

        train_losses.append(train_logs['loss'])
        val_losses.append(val_logs['loss'])
        test_losses.append(test_logs['loss'])

        plot_losses(train_losses, val_losses, test_losses, name, os.path.join(path, f'{name}/loss.png'))

    print('Training finished')
    
    return best_model


def test(model, test_dataloader, criterion, metrics, weight=None, device=torch.device('cuda:0')):
    test_logs = run(
        model=model, dataloader=test_dataloader,
        criterion=criterion, weight=weight, optimizer=None,
        metrics=metrics, phase='test', device = device
    )

    print('TEST | ', end='')
    report_metrics(test_logs)
    return test_logs
