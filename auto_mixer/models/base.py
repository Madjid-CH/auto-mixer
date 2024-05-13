from abc import ABC, abstractmethod, ABCMeta

import torch
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn

from auto_mixer.modules.train_test_module import AbstractTrainTestModule
from metrics import get_multiclass_metrics, get_multilabel_metrics


class Mixer(AbstractTrainTestModule, ABC):
    def __init__(self, target_length, optimizer_cfg, **kwargs):
        self.target_length = target_length
        super(Mixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.backbone = None
        self.classifier = None

    @abstractmethod
    def shared_step(self, batch, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_logits(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True, monitor='val_loss')

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class MulticlassMixer(Mixer, ABC):

    @abstractmethod
    def get_logits(self, images):
        pass

    def forward(self, x):
        return self.get_logits(x)

    def setup_criterion(self) -> torch.nn.Module:
        return nn.CrossEntropyLoss()

    def setup_scores(self):
        train_scores = get_multiclass_metrics(self.target_length)
        val_scores = get_multiclass_metrics(self.target_length)
        test_scores = get_multiclass_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]

    def predict(self, x, labels):
        logits = self.get_logits(x)
        loss = self.criterion(logits, labels.long())
        preds = torch.argmax(logits, dim=1)
        return {
            'preds': preds,
            'labels': labels,
            'loss': loss,
            'logits': logits
        }


class MultilabelMixer(Mixer, metaclass=ABCMeta):

    @abstractmethod
    def get_logits(self, images):
        pass

    def forward(self, x):
        return self.get_logits(x)

    def setup_criterion(self) -> torch.nn.Module:
        return nn.BCEWithLogitsLoss()

    def setup_scores(self):
        train_scores = get_multilabel_metrics(self.target_length)
        val_scores = get_multilabel_metrics(self.target_length)
        test_scores = get_multilabel_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]

    def predict(self, texts, labels):
        logits = self.get_logits(texts)
        loss = self.criterion(logits, labels.float())
        threshold = 0.5
        preds = torch.sigmoid(logits) > threshold
        preds = preds.float()
        return {
            'preds': preds,
            'labels': labels.long(),
            'loss': loss,
            'logits': logits
        }
