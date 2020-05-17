import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, average_precision_score

class Metric:
    def __init__(self, multiclass=False):
        self.reset()
        self.multiclass = multiclass
        self.summarizable = True

    def reset(self):
        self.outputs = None
        self.targets = None
        self.phase = 'test'

    def accumulate(self, output, target, phase='test'):
        self.phase = phase

        if self.outputs is None:
            self.outputs = output.detach().cpu()
            self.targets = target.cpu()
        else:
            self.outputs = torch.cat((self.outputs, output.detach().cpu()))
            self.targets = torch.cat((self.targets, target.cpu()))

    def __repr__(self):
        return f'{self.get()}'

class MultilabelMetric(Metric):
    def __init__(self, apply_sigmoid=True):
        super().__init__(multiclass=False)
        self.apply_sigmoid = apply_sigmoid
    
    def sigmoid(self, x):
        return torch.sigmoid(x) if self.apply_sigmoid else x

    def get_class_preds(self, class_idx):
        outputs = self.sigmoid(self.outputs)
        targets = self.targets

        class_outputs = outputs[:, class_idx]
        class_targets = targets[:, class_idx]
        
        return class_outputs, class_targets
    
    def get_thresholded_preds(self, class_idx, threshold):
        class_outputs, class_targets = self.get_class_preds(class_idx)
        class_outputs = (class_outputs > threshold).long()
        return class_outputs, class_targets

class Accuracy(Metric):
    __name__ = 'acc'

    def __init__(self, metric='acc', multiclass=False):
        super().__init__(multiclass=multiclass)
        self.metrics = {
            'acc': self.accuracy,
            'ba': self.ba,
        }

        self.__name__ = metric

    def get(self, threshold=0.5):
        return self.metrics[self.__name__](threshold=threshold)

    def accuracy(self, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()
        else:
            _, outputs = torch.max(outputs, 1)

        return accuracy_score(self.targets.numpy(), outputs.numpy())
    
    def ba(self, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()
        else:
            _, outputs = torch.max(outputs, 1)

        return balanced_accuracy_score(self.targets.numpy(), outputs.numpy())

    def get_best_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.targets.numpy(), self.outputs.numpy())
        best_idx = np.argmax(tpr-fpr)
        return thresholds[best_idx]

class MultilabelAccuracy(MultilabelMetric):
    __name__ = 'avg-ba'

    def __init__(self, metric='avg-ba', apply_sigmoid=True):
        super().__init__(apply_sigmoid=apply_sigmoid)

        self.metrics = {
            'top1-acc': self.top1_accuracy,
            'avg-acc': self.avg_accuracy,
            'avg-ba': self.avg_ba
        }

        self.__name__ = metric
    
    def get(self):
        return self.metrics[self.__name__]()
    
    def top1_accuracy(self):
        outputs = self.outputs
        _, outputs = torch.max(outputs, 1)
        _, targets = torch.max(self.targets, 1)
        return accuracy_score(targets, outputs.numpy())
    
    def top1_ba(self):
        outputs = self.outputs
        _, outputs = torch.max(outputs, 1)
        _, targets = torch.max(self.targets, 1)
        return balanced_accuracy_score(targets.numpy().astype('uint8'), outputs.numpy())
        
    def class_accuracy(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_thresholded_preds(class_idx, threshold)
        class_acuracy = accuracy_score(class_targets.numpy().astype('uint8'), class_outputs.numpy())
        
        return class_acuracy

    def avg_accuracy(self, threshold=[0.5]):
        if len(threshold) == 1:
            threshold = np.broadcast_to(threshold, self.targets.size(1))

        accuracy = [self.class_accuracy(idx, threshold[idx]) for idx in range(self.targets.size(1))]
        return np.mean(accuracy)

    def class_ba(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_thresholded_preds(class_idx, threshold)
        class_acuracy = balanced_accuracy_score(class_targets.numpy().astype('uint8'), class_outputs.numpy())
        
        return class_acuracy
    
    def avg_ba(self, threshold=[0.5]):
        if len(threshold) == 1:
            threshold = np.broadcast_to(threshold, self.targets.size(1))
        
        ba = [self.class_ba(idx, threshold[idx]) for idx in range(self.targets.size(1))]
        return np.mean(ba)
    
    def get_best_threshold(self, class_idx):
        class_outputs, class_targets = self.get_class_preds(class_idx)
        fpr, tpr, thresholds = roc_curve(class_targets.numpy(), class_outputs.numpy())
        best_idx = np.argmax(tpr-fpr)

        return thresholds[best_idx]
    
    def class_report(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_thresholded_preds(class_idx, threshold)
        report = classification_report(class_targets.numpy().astype('uint8'), class_outputs.numpy())
        
        return report

class RocAuc(Metric):
    __name__ = 'auc'

    def get(self):
        if self.multiclass:
            targets = np.zeros(self.outputs.shape)
            targets[torch.arange(self.targets.size(0)), self.targets] = 1.
            return roc_auc_score(targets.T, self.outputs.numpy().T, average='weighted', multi_class='ovr')
        return roc_auc_score(self.targets.numpy(), self.outputs.numpy())

    def get_curve(self):
        return roc_curve(self.targets.numpy(), self.outputs.numpy())

class MultilabelRocAuc(MultilabelMetric):
    __name__ = 'col-auc'
    
    def get(self):
        return self.avg_auc()
        
    def class_curve(self, class_idx):
        class_outputs, class_targets = self.get_class_preds(class_idx)
        return roc_curve(class_targets.numpy(), class_outputs.numpy())
    
    def class_auc(self, class_idx):
        class_outputs, class_targets = self.get_class_preds(class_idx)
        return roc_auc_score(class_targets.numpy(), class_outputs.numpy())
    
    def avg_auc(self):
        auc = [self.class_auc(class_idx) for class_idx in range(self.targets.size(1))]
        return np.mean(auc)

class Precision(Metric):
    __name__ = 'precision'

    def get(self, threshold=0.5):
        return self.precision(threshold=threshold)
    
    def precision(self, threshold):
        outputs = (self.outputs > threshold).long()
        return precision_score(self.targets.numpy(), outputs.numpy())

    def avg_precision(self):
        return average_precision_score(self.targets.numpy(), self.outputs.numpy())

    def get_curve(self):
        return precision_recall_curve(self.targets.numpy(), self.outputs.numpy())



class FScore(Metric):
    __name__ = 'f-score'

    def get(self, threshold=0.5):
        if self.multiclass:
            _, outputs = torch.max(self.outputs, 1)
            return f1_score(self.targets.numpy(), outputs.numpy(), average='weighted')
        outputs = (self.outputs > threshold).long()
        return f1_score(self.targets.numpy(), outputs.numpy())

class ConfusionMatrix(Metric):
    __name__ = 'cm'

    def __init__(self, multiclass=False):
        super().__init__(multiclass=multiclass)
        self.summarizable = False

    def get(self, normalized=False, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()
        else:
            _, outputs = torch.max(self.outputs, 1)

        cm = confusion_matrix(self.targets.numpy(), outputs.numpy())

        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        return cm

class MultilabelConfusionMatrix(MultilabelMetric):
    __name__ = 'cm'

    def __init__(self):
        super().__init__(apply_sigmoid=False)
        self.summarizable = False

    def get(self, normalized=False):
        return self.top1_cm(normalized=normalized)
    
    def top1_cm(self, normalized=False):
        outputs = self.outputs
        targets = self.targets
        _, outputs = torch.max(self.outputs, 1)
        _, targets = torch.max(self.targets, 1)
       
        cm = confusion_matrix(targets.numpy(), outputs.numpy())

        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        return cm

    def class_cm(self, class_idx, normalized=False, threshold=0.5):
        class_outputs, class_targets = self.get_thresholded_preds(class_idx, threshold=threshold)
        cm = confusion_matrix(class_targets.numpy(), class_outputs.numpy())
        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
        return cm