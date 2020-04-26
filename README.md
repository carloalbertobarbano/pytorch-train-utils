# pytorch-train-utils

## Install

```bash
pip3 install git+git://github.com/carloalbertobarbano/pytorch-train-utils
```

## Example

```python
from pytorchtrainutils import trainer
from pytorchtrainutils import metrics
from pytorchtrainutils import utils

. . .

utils.set_seed(42)
mean, std = utils.get_mean_and_std(train_loader)

. . .

tracked_metrics = [
    metrics.Accuracy(multiclass=True),
    metrics.FScore(multiclass=True),
    metrics.RocAuc(multiclass=True),
    metrics.ConfusionMatrix(multiclass=True)
]

name = 'runs/test'
best_model = trainer.fit(
    model, train_dataloader=train_loader, val_dataloader=val_loader,
    test_dataloader=test_loader, test_every=2, criterion=criterion,
    optimizer=optimizer, scheduler=None, metrics=tracked_metrics, n_epochs=20,
    name=name, device=device,
    callbacks={'train': lambda: utils.save_cm(
        cm=tracked_metrics[-1], title='train', path=f'{name}/cm-train.png',
        normalized=True, format=".1f", vmin=0., vmax=1.,
        yticklabels=['a']*10)
    }
)

test_logs = trainer.test(
    best_model, test_dataloader=test_loader,
    criterion=criterion, metrics=tracked_metrics,
    device=device
)
```
