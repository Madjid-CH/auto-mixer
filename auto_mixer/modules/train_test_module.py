import abc
import time
from typing import List, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Metric


class AbstractTrainTestModule(pl.LightningModule, abc.ABC):
    def __init__(self, optimizer_cfg: DictConfig, log_confusion_matrix: bool = False, **kwargs):
        self.optimizer_cfg = optimizer_cfg
        self.log_confusion_matrix = log_confusion_matrix
        self.loss_pos_weight = torch.tensor(optimizer_cfg.loss_pos_weight) if 'loss_pos_weight' in optimizer_cfg \
            else None
        if 'loss_pos_weight' in optimizer_cfg:
            self.optimizer_cfg.pop('loss_pos_weight')

        super(AbstractTrainTestModule, self).__init__(**kwargs)
        self.criterion = self.setup_criterion()
        self.train_scores, self.val_scores, self.test_scores = self.setup_scores()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        if any([isinstance(self.train_scores, list), isinstance(self.val_scores, list),
                isinstance(self.test_scores, list)]):
            raise ValueError('Scores must be a dict')
        self.best_epochs = {'val_loss': None}
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.best_epochs[metric] = None

        self.logged_n_parameters = False
        self.train_time_start = None
        self.test_time_start = None

    def on_train_start(self) -> None:
        super().on_train_start()
        self.train_time_start = time.time()

    def on_test_start(self) -> None:
        super().on_test_start()
        self.test_time_start = time.time()

    def on_test_end(self) -> None:
        super().on_test_end()
        test_time = time.time() - self.test_time_start
        if self.logger:
            self.logger.experiment.add_scalar("test_time", test_time, global_step=self.current_epoch)

    @abc.abstractmethod
    def setup_criterion(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def setup_scores(self) -> List[Dict[str, Metric]]:
        raise NotImplementedError

    @abc.abstractmethod
    def shared_step(self, batch, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


    def training_step(self, batch, batch_idx):
        if self.train_scores is not None:
            for metric in self.train_scores:
                self.train_scores[metric].to(self.device)
        results = self.shared_step(batch, mode='train')
        f = 'loss' if 'loss' in results else 'loss_fusion'
        self.log('', results[f], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.train_scores is not None:
            for metric in self.train_scores:
                score = self.train_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
                self.log(f'train_{metric}_step', score, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                         sync_dist=True)
        self.log('train_loss_step', results[f], on_step=True, on_epoch=False, prog_bar=True, logger=True, )
        self.training_step_outputs.append(results)
        return results

    def on_train_epoch_end(self):
        self.log_training_metrics()

    def log_training_metrics(self):
        if self.train_scores is not None:
            for metric in self.train_scores:
                train_score = self.train_scores[metric].compute()
                self.log(f'train_{metric}', train_score, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', np.mean([output['loss'].cpu().item() for output in self.training_step_outputs]),
                 prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.val_scores[metric].to(self.device)
        results = self.shared_step(batch, mode='val')
        r = 'loss' if 'loss' in results else 'loss_fusion'
        self.log('val_loss', results[r], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.val_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        self.log_validation_metrics()

    def log_validation_metrics(self):
        if self.val_scores is not None:
            for metric in self.val_scores:
                val_score = self.val_scores[metric].compute()
                self.log(f'val_{metric}', val_score, prog_bar=True, logger=True)
        f = 'loss' if 'loss' in self.validation_step_outputs[0] else 'loss_fusion'
        val_loss = np.mean([output[f].cpu().item() for output in self.validation_step_outputs])
        self.log('val_loss', val_loss, prog_bar=True, logger=True, sync_dist=True)

        if self.best_epochs['val_loss'] is None or (val_loss <= self.best_epochs['val_loss'][1]):
            self.best_epochs['val_loss'] = (self.current_epoch, val_loss)
            self.log('best_val_loss', val_loss, prog_bar=True, logger=True, sync_dist=True)
            self.log('best_val_loss_epoch', self.current_epoch, prog_bar=True, logger=True, sync_dist=True)
            if self.train_time_start is not None:
                duration = time.time() - self.train_time_start
                self.log('best_val_loss_time', duration, prog_bar=True, logger=True, sync_dist=True)
            if self.val_scores is not None:
                for metric in self.val_scores:
                    val_score = self.val_scores[metric].compute()
                    self.log(f'best_val_{metric}', val_score, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        if self.test_scores is not None:
            for metric in self.test_scores:
                self.test_scores[metric].to(self.device)
        results = self.shared_step(batch, mode='test')
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.test_scores is not None:
            for metric in self.test_scores:
                self.test_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
        self.test_step_outputs.append(results)
        return results

    def on_test_epoch_end(self):
        self.log_test_metrics()
        self.test_step_outputs.clear()

    def log_test_metrics(self):
        if self.test_scores is not None:
            for metric in self.test_scores:
                test_score = self.test_scores[metric].compute()
                self.log(f'test_{metric}', test_score, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        results = self.shared_step(batch)
        return results

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
