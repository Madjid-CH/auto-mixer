from typing import Literal

import torchmetrics as tm

_Task = Literal["binary", "multiclass", "multilabel"]


def get_multiclass_metrics(num_classes):
    t: _Task = 'multiclass'
    return dict(
        accuracy=tm.Accuracy(task=t, num_classes=num_classes),
        precision_macro=tm.Precision(task=t, num_classes=num_classes, average='macro'),
        precision_micro=tm.Precision(task=t, num_classes=num_classes, average='micro'),
        f1_macro=tm.F1Score(task=t, num_classes=num_classes, average='macro'),
        f1_micro=tm.F1Score(task=t, num_classes=num_classes, average='micro'),
        f1_weighted=tm.F1Score(task=t, num_classes=num_classes, average='weighted'),
        recall_macro=tm.Recall(task=t, num_classes=num_classes, average='macro'),
        recall_micro=tm.Recall(task=t, num_classes=num_classes, average='micro'),
    )


def get_multilabel_metrics(num_labels):
    t: _Task = 'multilabel'
    return dict(
        accuracy=tm.Accuracy(task=t, num_labels=num_labels),
        auroc_macro=tm.AUROC(task=t, num_labels=num_labels, average='macro'),
        auroc_weighted=tm.AUROC(task=t, num_labels=num_labels, average='weighted'),
        precision_macro=tm.Precision(task=t, num_labels=num_labels, average='macro'),
        precision_micro=tm.Precision(task=t, num_labels=num_labels, average='micro'),
        f1_macro=tm.F1Score(task=t, num_labels=num_labels, average='macro'),
        f1_micro=tm.F1Score(task=t, num_labels=num_labels, average='micro'),
        f1_weighted=tm.F1Score(task=t, num_labels=num_labels, average='weighted'),
        recall_macro=tm.Recall(task=t, num_labels=num_labels, average='macro'),
        recall_micro=tm.Recall(task=t, num_labels=num_labels, average='micro'),
    )


class SelectiveAUROC:
    def __init__(self, selected_indices, **kwargs):
        self.selected_indices = selected_indices
        self.auroc = tm.AUROC(**kwargs)

    def update(self, preds, target):
        preds = preds[:, self.selected_indices]
        target = target[:, self.selected_indices]
        self.auroc.update(preds, target)

    def compute(self):
        return self.auroc.compute()

    def to(self, device):
        self.auroc.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return self.compute()
