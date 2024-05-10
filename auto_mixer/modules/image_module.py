from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn

from auto_mixer import modules
from auto_mixer.metrics import get_multilabel_metrics, get_multiclass_metrics
from auto_mixer.modules import TwoLayeredPerceptron
from auto_mixer.modules.train_test_module import AbstractTrainTestModule


class ImageMixer(AbstractTrainTestModule, ABC):
    def __init__(self, target_length, optimizer_cfg, **kwargs):
        super(ImageMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.target_length = target_length
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


class MultilabelImageMixer(ImageMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(ImageMixer, self).__init__(optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.modalities.block_type)
        self.backbone = model_cls(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = TwoLayeredPerceptron(input_dim=512, hidden_dim=512, output_dim=self.target_length)

    def shared_step(self, batch, **kwargs):
        images, labels = batch
        image_logits = self.get_logits(images)
        loss = self.criterion(image_logits, labels.float())
        threshold = 0.5
        preds = torch.sigmoid(image_logits) > threshold
        preds = preds.float()
        return {
            'preds': preds,
            'labels': labels.long(),
            'loss': loss,
            'logits': image_logits
        }

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits

    def forward(self, x):
        return self.get_logits(x)

    def setup_criterion(self) -> torch.nn.Module:
        return nn.BCEWithLogitsLoss()

    def setup_scores(self):
        train_scores = get_multilabel_metrics(self.target_length)
        val_scores = get_multilabel_metrics(self.target_length)
        test_scores = get_multilabel_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]


class MulticlassImageMixer(ImageMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(ImageMixer, self).__init__(optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.modalities.block_type)
        self.backbone = model_cls(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = TwoLayeredPerceptron(input_dim=512, hidden_dim=512, output_dim=self.target_length)

    def shared_step(self, batch, **kwargs):
        images, labels = batch
        image_logits = self.get_logits(images)
        loss = self.criterion(image_logits, labels.long())
        preds = torch.argmax(image_logits, dim=1)
        return {
            'preds': preds,
            'labels': labels,
            'loss': loss,
            'logits': image_logits
        }

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits

    def forward(self, x):
        return self.get_logits(x)

    def setup_criterion(self) -> torch.nn.Module:
        return nn.CrossEntropyLoss()

    def setup_scores(self):
        train_scores = get_multiclass_metrics(self.target_length)
        val_scores = get_multiclass_metrics(self.target_length)
        test_scores = get_multiclass_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]
