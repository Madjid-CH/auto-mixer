import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from auto_mixer import modules
from auto_mixer.metrics import get_multilabel_metrics, get_multiclass_metrics
from auto_mixer.modules import StandardClassifier
from auto_mixer.modules.train_test_module import AbstractTrainTestModule


class MultiLabelMultiLoss(AbstractTrainTestModule):
    def __init__(self, encoders, target_length, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.target_length = target_length
        super(MultiLabelMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        multimodal_config = model_cfg.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.encoders = MultiLabelMultiLoss._build_encoders(encoders)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.multimodal)

        num_patches = self.fusion_function.get_output_shape(*[v.num_patch for v in self.encoders.values()], dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifiers = self._build_classifiers(target_length)
        self.classifier_fusion = StandardClassifier(input_shape=(16, 49, 729), num_classes=target_length)

        self.criteria = {k: nn.BCEWithLogitsLoss() for k in self.encoders.keys()}
        self.fusion_criterion = nn.BCEWithLogitsLoss()
        self.fusion_loss_weight = 1.0 / (len(self.encoders) + 1)
        self.fusion_loss_change = 0
        self.loss_change_epoch = 0

    def _build_classifiers(self, target_length):
        return {'images': nn.Linear(in_features=512, out_features=23, bias=True).cuda(),
                'texts': nn.Linear(in_features=729, out_features=23, bias=True).cuda()}

    @staticmethod
    def _build_encoders(encoders):
        return {k: v[1].cuda() for k, v in encoders.items()}

    def shared_step(self, batch, **kwargs):
        labels = torch.tensor(batch['labels']).float().cuda()
        logits = {
            k: v(batch[k]) for k, v in self.encoders.items()
        }
        fused_modalities = self.fusion_function(*list(logits.values()))
        fused_logits = self.fusion_mixer(fused_modalities)

        logits = {k: v.reshape(v.shape[0], -1, v.shape[-1]) for k, v in logits.items()}
        logits = {k: v.mean(dim=1) for k, v in logits.items()}
        logits = {k: self.classifiers[k](v) for k, v in logits.items()}
        fused_logits = self.classifier_fusion(fused_logits)

        losses = {
            k: self.criteria[k](v, labels) for k, v in logits.items()
        }

        loss = self._compute_multi_loss(fused_logits, labels, losses)

        threshold = 0.5
        preds = torch.sigmoid(fused_logits) > threshold
        preds = preds.float()
        modalities_preds = {k: torch.sigmoid(v) > threshold for k, v in logits.items()}
        modalities_preds = {k: v.float() for k, v in modalities_preds.items()}

        return {
            'preds': preds,
            'modalities_preds': modalities_preds,
            'labels': labels.long(),
            'loss': loss,
            'losses': losses,
            'logits': logits
        }

    def _compute_multi_loss(self, fused_logits, labels, losses):
        loss_fusion = self.fusion_criterion(fused_logits, labels)
        loss = self.fusion_loss_weight * loss_fusion
        ow = (1 - self.fusion_loss_weight) / len(self.encoders)
        loss += sum([ow * v for v in losses.values()])
        loss *= (len(self.encoders) + 1)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_training_metrics()
        for k in self.encoders.keys():
            wandb.log(
                {f'train_loss_{k}': torch.stack([x['losses'][k] for x in self.training_step_outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss'] for x in self.training_step_outputs]).mean().item(),
                 sync_dist=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.log_validation_metrics()
        val_loss_fusion = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion, sync_dist=True)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
        self.validation_step_outputs.clear()

    def setup_criterion(self) -> None:
        return None

    def setup_scores(self):
        train_scores = get_multilabel_metrics(self.target_length)
        val_scores = get_multilabel_metrics(self.target_length)
        test_scores = get_multilabel_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class MultiClassMultiLoss(AbstractTrainTestModule):
    def __init__(self, encoders, target_length, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.target_length = target_length
        super(MultiClassMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        multimodal_config = model_cfg.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.encoders = MultiClassMultiLoss._build_encoders(encoders)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.multimodal)
        num_patches = self.fusion_function.get_output_shape(*[v.num_patch for v in self.encoders.values()], dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifiers = {k: torch.nn.Linear(v.hidden_dim, target_length)
                            for k, v in self.encoders.items()}
        self.classifier_fusion = StandardClassifier(input_shape=(16, 49, 256), num_classes=target_length)

        self.criteria = {k: nn.CrossEntropyLoss() for k in self.encoders.keys()}
        self.fusion_criterion = nn.CrossEntropyLoss()
        self.fusion_loss_weight = 1.0 / (len(self.encoders) + 1)
        self.fusion_loss_change = 0
        self.loss_change_epoch = 0

    @staticmethod
    def _build_encoders(encoders):
        return {k: v[1] for k, v in encoders.items()}

    def shared_step(self, batch, **kwargs):
        labels = torch.tensor(batch['labels']).float().cuda()

        logits = {
            k: v(batch[k]) for k, v in self.encoders.items()
        }

        fused_modalities = self.fusion_function(*list(logits.values()))
        fused_logits = self.fusion_mixer(fused_modalities)

        logits = {k: v.reshape(v.shape[0], -1, v.shape[-1]) for k, v in logits.items()}
        logits = {k: v.mean(dim=1) for k, v in logits.items()}
        logits = {k: self.classifiers[k](v) for k, v in logits.items()}
        fused_logits = self.classifier_fusion(fused_logits)

        # compute losses
        losses = {
            k: self.criteria[k](v, labels) for k, v in logits.items()
        }

        loss = self._compute_multi_loss(fused_logits, labels, losses)

        preds = torch.softmax(fused_logits, dim=1).argmax(dim=1)
        modalities_preds = {k: torch.softmax(v, dim=1).argmax(dim=1) for k, v in logits.items()}

        return {
            'preds': preds,
            'modalities_preds': modalities_preds,
            'labels': labels,
            'loss_fusion': loss,
            'losses': losses,
            'logits': logits
        }

    def _compute_multi_loss(self, fused_logits, labels, losses):
        loss_fusion = self.fusion_criterion(fused_logits, labels)
        loss = self.fusion_loss_weight * loss_fusion
        ow = (1 - self.fusion_loss_weight) / len(self.encoders)
        loss += sum([ow * v for v in losses.values()])
        loss *= (len(self.encoders) + 1)
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        for k in self.encoders.keys():
            wandb.log(
                {f'train_loss_{k}': torch.stack([x['losses'][k] for x in self.training_step_outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in self.training_step_outputs]).mean().item(),
                 sync_dist=True)

    def on_validation_epoch_end(self, ) -> None:
        super().validation_epoch_end()
        val_loss_fusion = torch.stack([x['loss_fusion'] for x in self.validation_step_outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion, sync_dist=True)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)

    def setup_criterion(self) -> None:
        return None

    def setup_scores(self):
        train_scores = get_multiclass_metrics(self.target_length)
        val_scores = get_multiclass_metrics(self.target_length)
        test_scores = get_multiclass_metrics(self.target_length)
        return [train_scores, val_scores, test_scores]

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
