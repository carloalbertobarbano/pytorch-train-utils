import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

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

class Accuracy(Metric):
    __name__ = 'acc'

    def get(self, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()
        else:
            _, outputs = torch.max(outputs, 1)

        return accuracy_score(self.targets.numpy(), outputs.numpy())

    def get_best_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.targets.numpy(), self.outputs.numpy())
        best_idx = np.argmax(tpr-fpr)
        return thresholds[best_idx]

class MultilabelAccuracy(Metric):
    __name__ = 'acc'
    
    def get(self, threshold=0.5):
        outputs = self.outputs
        _, outputs = torch.max(outputs, 1)
        _, targets = torch.max(self.targets, 1)
        return accuracy_score(targets, outputs.numpy())
    
    def class_avg(self, threshold=[0.5]):
        if len(threshold) == 1:
            threshold = np.broadcast_to(threshold, self.targets.size(1))

        outputs = torch.sigmoid(self.outputs)
        outputs = (outputs > torch.tensor(threshold)).long()
        targets = self.targets

        accuracy = []
        for class_idx in range(targets.size(1)):
            class_outputs = outputs[:, class_idx]
            class_targets = targets[:, class_idx]

            class_acuracy = accuracy_score(class_targets.numpy().astype('uint8'), class_outputs.numpy())
            accuracy.append(class_acuracy)
        
        accuracy = np.array(accuracy)
        return accuracy.mean()
    
    def get_best_threshold(self, class_idx):
        outputs = torch.sigmoid(self.outputs)
        targets = self.targets

        class_outputs = outputs[:, class_idx]
        class_targets = targets[:, class_idx]

        fpr, tpr, thresholds = roc_curve(class_targets.numpy(), class_outputs.numpy())
        best_idx = np.argmax(tpr-fpr)
        return thresholds[best_idx]

    def get_class_preds(self, class_idx, threshold=0.5):
        outputs = torch.sigmoid(self.outputs)
        outputs = (outputs > torch.tensor(threshold)).long()
        targets = self.targets

        
        class_outputs = outputs[:, class_idx]
        class_targets = targets[:, class_idx]
        
        return class_outputs, class_targets
    
    def class_accuracy(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_class_preds(class_idx, threshold)
        class_acuracy = accuracy_score(class_targets.numpy().astype('uint8'), class_outputs.numpy())
        
        return class_acuracy
    
    def class_ba(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_class_preds(class_idx, threshold)
        class_acuracy = balanced_accuracy_score(class_targets.numpy().astype('uint8'), class_outputs.numpy())
        
        return class_acuracy
    
    def class_report(self, class_idx, threshold=0.5):
        class_outputs, class_targets = self.get_class_preds(class_idx, threshold)
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

class MultilabelRocAuc(Metric):
    __name__ = 'col-auc'
    
    def get_class_preds(self, class_idx):
        outputs = torch.sigmoid(self.outputs)
        targets = self.targets

        class_outputs = outputs[:, class_idx]
        class_targets = targets[:, class_idx]

        return class_outputs, class_targets
    
    def get(self):
        roc = []
        for class_idx in range(self.targets.size(1)):            
            class_outputs, class_targets = self.get_class_preds(class_idx)
            class_roc = roc_auc_score(class_targets.numpy(), class_outputs.numpy())
            roc.append(class_roc)
                
        return np.mean(roc)
        
    def get_curve(self, class_idx):
        class_outputs,class_targets = self.get_class_preds(class_idx)
        return roc_curve(class_targets.numpy(), class_outputs.numpy())

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

class MultilabelConfusionMatrix(Metric):
    __name__ = 'cm'

    def __init__(self):
        super().__init__(multiclass=False)
        self.summarizable = False

    def get(self, normalized=False):
        outputs = self.outputs
        targets = self.targets
        _, outputs = torch.max(self.outputs, 1)
        _, targets = torch.max(self.targets, 1)
       
        cm = confusion_matrix(targets.numpy(), outputs.numpy())

        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        return cm
    
    def get_class_preds(self, class_idx, threshold=0.5):
        outputs = torch.sigmoid(self.outputs)
        outputs = (outputs > torch.tensor(threshold)).long()
        targets = self.targets

        
        class_outputs = outputs[:, class_idx]
        class_targets = targets[:, class_idx]
        
        return class_outputs, class_targets
    
    def get_class(self, class_idx, normalized=False, threshold=0.5):
        class_outputs, class_targets = self.get_class_preds(class_idx, threshold=threshold)
        cm = confusion_matrix(class_targets.numpy(), class_outputs.numpy())
        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
        return cm