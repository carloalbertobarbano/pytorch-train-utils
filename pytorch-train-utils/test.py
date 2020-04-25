import torch
import torch.nn.functional as F
import torchvision
import trainer
import metrics
import utils

device = torch.device('cpu')
utils.set_seed(42)

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

D_in, H, D_out =  1000, 100, 10
train_x, train_y = torch.randn(100, D_in), torch.max(torch.randn(100, D_out), 1)[1]
val_x, val_y = torch.randn(20, D_in), torch.max(torch.randn(20, D_out), 1)[1]
test_x, test_y = torch.randn(30, D_in), torch.max(torch.randn(30, D_out), 1)[1]


train_dataset = Dataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, num_workers=0, shuffle=True)
val_dataset = Dataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=0, shuffle=False)
test_dataset = Dataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0, shuffle=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax(dim=1)
)

criterion = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)

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
        yticklabels=['a']*10
    )})

test_logs = trainer.test(
    best_model, test_dataloader=test_loader,
    criterion=criterion, metrics=tracked_metrics,
    device=device
)
