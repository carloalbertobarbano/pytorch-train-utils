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

def update_df(df, epoch, metrics):
    summarizable = dict(filter(lambda m: m[1].summarizable if hasattr(m[1], 'summarizable') else True, metrics.items()))
    summarizable = {k: float(str(v)) for k,v in summarizable.items()}
    summarizable.update({'epoch': epoch})

    df2 = pd.DataFrame([summarizable]).set_index('epoch')
    return pd.concat((df, df2)).apply(pd.to_numeric, errors='ignore')

def run(model, dataloader, criterion, optimizer, metrics, phase, 
        device=torch.device('cuda:0'), weight=None, tta=False, silence=False,
        accumulation_steps=1):
    num_batches = 0.
    loss = 0.

    if phase == 'train':
        model.train()
    else:
        model.eval()

    for metric in metrics:
        metric.reset()

    itr = None
    if silence:
        itr = iter(dataloader)
    if not silence:
        itr = iter(tqdm(dataloader, desc=phase, leave=False))
        
    for step, (data, labels) in enumerate(itr):
        data, labels = data.to(device), labels.to(device)

        running_loss = 0.
        output = None

        with torch.set_grad_enabled(phase == 'train'):
            if tta:
                batch_size, n_crops, c, h, w = data.size()
                data = data.view(-1, c, h, w)
                output = model(data)
                output = output.view(batch_size, n_crops, -1).mean(1)
            else:
                output = model(data)

            for metric in metrics:
                metric.accumulate(output.clone(), labels.clone())

            running_loss = criterion(output, labels, weight=weight)

        if phase == 'train':
            running_loss.backward()

            if (step + 1) % accumulation_steps == 0:
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.step()
                else:
                    for opt in optimizer:
                        opt.step()

                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.zero_grad()
                else:
                    for opt in optimizer:
                        opt.zero_grad()

        loss += running_loss.item()
        
        if (step + 1) % accumulation_steps == 0:
            num_batches += 1

    logs = { metric.__name__: copy.deepcopy(metric) for metric in metrics }
    logs.update({'loss': loss / num_batches})
    return logs

def make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params=None):
    checkpoint = {
        'epoch': epoch, 'model': model.state_dict()
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
    plt.close(hm)

def plot_metrics(df, path, name):
    ax = df.drop(['loss'], axis=1).plot()
    plt.title(name)
    plt.savefig(path)
    plt.close(ax.get_figure())

def fit(model, train_dataloader, val_dataloader, test_dataloader, test_every,
        criterion, optimizer, scheduler, metrics, n_epochs, name, path='',
        weight={'train': None, 'val': None, 'test': None},
        metric_choice='loss', mode='min', device=torch.device('cuda:0'), checkpoint_params=None, 
        callbacks={'train': None, 'val': None, 'test':None}, silence=False, accumulation_steps=1):
    utils.ensure_dir(name)

    best_metric = 0.
    best_model = None

    train_losses = []
    val_losses = []
    test_losses = []

    test_logs = {'loss': 1.}

    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()

    for epoch in range(n_epochs):

        train_logs = run(
            model=model, dataloader=train_dataloader,
            criterion=criterion, weight=weight['train'], optimizer=optimizer,
            metrics=metrics, phase='train', device=device, silence=silence, 
            accumulation_steps=accumulation_steps
        )

        if 'train' in callbacks and callbacks['train'] is not None:
            callbacks['train']()

        val_logs = run(
            model=model, dataloader=val_dataloader,
            criterion=criterion, weight=weight['val'], optimizer=None,
            metrics=metrics, phase='val', device=device, silence=silence
        )

        if 'val' in callbacks and callbacks['val'] is not None:
            callbacks['val']()

        if not silence:
            print(f'Epoch: {epoch:03d} | VAL ', end='')
            report_metrics(val_logs, end=' | TRAIN ')
            report_metrics(train_logs, end=' |\n')

            df_train = update_df(df_train, epoch, train_logs)
            df_valid = update_df(df_valid, epoch, val_logs)
            df_train.to_csv(f'{name}/metrics-train.csv')
            df_valid.to_csv(f'{name}/metrics-val.csv')

            plot_metrics(df_train, f'{name}/metrics-train.png', name)
            plot_metrics(df_valid, f'{name}/metrics-val.png', name)

            torch.save(
                make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params),
                os.path.join(path, f'{name}/final.pt')
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_logs['loss'])
            else:
                scheduler.step()
           
        if best_model is None or is_better(float(str(val_logs[metric_choice])), best_metric, mode):
            best_metric = float(str(val_logs[metric_choice]))
            best_model = copy.deepcopy(model)
            if not silence:
                torch.save(
                    make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params),
                    os.path.join(f'{name}/best.pt')
                )

        if test_dataloader is not None and (epoch+1) % test_every == 0:
            test_logs = test(
                model=model, test_dataloader=test_dataloader,
                criterion=criterion, metrics=metrics,
                device=device,
                weight=weight['test'],
                silence=silence
            )

            if 'test' in callbacks and callbacks['test'] is not None:
                callbacks['test']()

        if not silence:
            train_losses.append(train_logs['loss'])
            val_losses.append(val_logs['loss'])
            test_losses.append(test_logs['loss'])

            plot_losses(train_losses, val_losses, test_losses, name, os.path.join(path, f'{name}/loss.png'))

    if not silence:
        print('Training finished')

    return best_model


def test(model, test_dataloader, criterion, metrics, weight=None, device=torch.device('cuda:0'), tta=False, silence=False):
    test_logs = run(
        model=model, dataloader=test_dataloader,
        criterion=criterion, weight=weight, optimizer=None,
        metrics=metrics, phase='test', device=device,
        tta=tta
    )

    if not silence:
        print('TEST | ', end='')
        report_metrics(test_logs)
    return test_logs
