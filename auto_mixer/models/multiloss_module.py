import os
from typing import Any

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from auto_mixer import modules
from auto_mixer.modules.train_test_module import AbstractTrainTestModule


class MulticlassMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MulticlassMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        self.freeze_modalities_on_epoch = model_cfg.get('freeze_modalities_on_epoch', None)
        self.random_modality_muting_on_freeze = model_cfg.get('random_modality_muting_on_freeze', False)
        self.muting_probs = model_cfg.get('muting_probs', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(**audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)

        self.image_criterion = nn.CrossEntropyLoss()
        self.audio_criterion = nn.CrossEntropyLoss()
        self.fusion_criterion = nn.CrossEntropyLoss()
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.loss_change_epoch = model_cfg.get('loss_change_epoch', 0)

    def shared_step(self, batch, **kwargs):

        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train':
            if self.freeze_modalities_on_epoch is not None and (self.current_epoch == self.freeze_modalities_on_epoch) \
                    and not self.modalities_freezed:
                self._freeze_modalities()
            if self.random_modality_muting_on_freeze and (self.current_epoch >= self.freeze_modalities_on_epoch):
                self.mute = np.random.choice(['image', 'audio', 'multimodal'], p=[self.muting_probs['image'],
                                                                                  self.muting_probs['audio'],
                                                                                  self.muting_probs['multimodal']])

            if self.mute != 'multimodal':
                if self.mute == 'image':
                    image = torch.zeros_like(image)
                elif self.mute == 'audio':
                    audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)
        loss_fusion = self.fusion_criterion(logits, labels)

        ow = (1 - self.fusion_loss_weight) / 2
        loss = (self.fusion_loss_weight * loss_fusion + ow * loss_image + ow * loss_audio) * 3
        if self.modalities_freezed and kwargs.get('mode', None) == 'train':
            loss = loss_fusion

        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        preds_image = torch.softmax(image_logits, dim=1).argmax(dim=1)
        preds_audio = torch.softmax(audio_logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'preds_image': preds_image,
            'preds_audio': preds_audio,
            'labels': labels,
            'loss': loss,
            'loss_image': loss_image,
            'loss_audio': loss_audio,
            'loss_fusion': loss_fusion,
            'image_logits': image_logits,
            'audio_logits': audio_logits,
            'logits': logits
        }

    def _freeze_modalities(self):
        print('Freezing modalities')
        for param in self.image_mixer.parameters():
            param.requires_grad = False
        for param in self.audio_mixer.parameters():
            param.requires_grad = False
        for param in self.classifier_image.parameters():
            param.requires_grad = False
        for param in self.classifier_audio.parameters():
            param.requires_grad = False
        self.modalities_freezed = True

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        wandb.log({'train_loss_image': torch.stack([x['loss_image'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_audio': torch.stack([x['loss_audio'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack([x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in outputs]).mean().item(), sync_dist=True)

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        val_loss_fusion = torch.stack([x['loss_fusion'] for x in outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion, sync_dist=True)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
        if self.use_softadapt:
            self.image_criterion_history.append(torch.stack([x['loss_image'] for x in outputs]).mean().item())
            self.audio_criterion_history.append(torch.stack([x['loss_audio'] for x in outputs]).mean().item())
            self.fusion_criterion_history.append(torch.stack([x['loss_fusion'] for x in outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_audio': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
            wandb.log({'val_loss_image': self.image_criterion_history[-1]})
            wandb.log({'val_loss_audio': self.audio_criterion_history[-1]})
            wandb.log({'val_loss_fusion': self.fusion_criterion_history[-1]})
            self.log('val_loss_fusion', self.fusion_criterion_history[-1], sync_dist=True)

            if self.current_epoch != 0 and (self.current_epoch % self.update_loss_weights_per_epoch == 0):
                print('[!] Updating loss weights')
                self.loss_weights = self.softadapt.get_component_weights(torch.tensor(self.image_criterion_history),
                                                                         torch.tensor(self.audio_criterion_history),
                                                                         torch.tensor(self.fusion_criterion_history),
                                                                         verbose=True)
                print(f'[!] loss weights: {self.loss_weights}')
                self.image_criterion_history = list()
                self.audio_criterion_history = list()
                self.fusion_criterion_history = list()

    def setup_criterion(self) -> None:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                            f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                            prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                            rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                          f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                          prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                          rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                           f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                           prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                           rec_m=Recall(task="multiclass", num_classes=10, average='macro'))

        return [train_scores, val_scores, test_scores]

    def test_epoch_end(self, outputs, save_preds=False):
        super().test_epoch_end(outputs, save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        preds_image = torch.cat([x['preds_image'] for x in outputs])
        preds_audio = torch.cat([x['preds_audio'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        image_logits = torch.cat([x['image_logits'] for x in outputs])
        audio_logits = torch.cat([x['audio_logits'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])

        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = os.path.dirname(self.checkpoint_path)
        torch.save(dict(preds=preds, preds_image=preds_image, preds_audio=preds_audio, labels=labels,
                        image_logits=image_logits, audio_logits=audio_logits, logits=logits),
                   save_path + '/test_preds.pt')
        print(f'[!] Saved test predictions to {save_path}/test_preds.pt')

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def intermediate_step(self, batch: Any) -> Any:
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # fuse modalities
        results = self.shared_step(batch)

        fusion_correct = results['preds'] == labels
        image_correct = results['preds_image'] == labels
        audio_correct = results['preds_audio'] == labels

        return dict(image_logits=image_logits, audio_logits=audio_logits, logits=logits,
                    fusion_correct=fusion_correct, image_correct=image_correct, audio_correct=audio_correct)
