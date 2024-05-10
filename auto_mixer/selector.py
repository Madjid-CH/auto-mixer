import os

import torch
from omegaconf import OmegaConf

from auto_mixer.modules.image_module import MulticlassImageMixer, MultilabelImageMixer
import pytorch_lightning as pl


def select_encoder_for(modality, task, train_dataloader, val_dataloader):
    selector = selectors[modality]
    encoder = selector(task, train_dataloader, val_dataloader)
    return encoder


def select_image_encoder(task, train_dataloader, val_dataloader):
    modules_configs_files = os.listdir("auto_mixer/cfg/image_models")
    cfgs = [OmegaConf.load(cfg_file) for cfg_file in modules_configs_files]
    train_cfg = OmegaConf.load("auto_mixer/cfg/micro_train.yml")
    Mixer = get_model_for(task)
    target_length = len(train_dataloader.dataset[0]['labels'])
    models = {
        cfg.block_type: Mixer(
            target_length=target_length, model_cfg=cfg, optimizer_cfg=train_cfg.optimzer
        ) for cfg in cfgs
    }
    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        log_every_n_steps=train_cfg.log_interval_steps,
        max_epochs=train_cfg.epochs
    )

    for model in models.values():
        trainer.fit(model, train_dataloader, val_dataloader)
        results = trainer.test(model, val_dataloader)
        model.results = results

    block_type, best_model = max(models.items(), key=lambda x: x[1].results['test_accuracy'])
    return block_type, best_model.backbone


def get_model_for(task):
    if task == "multiclass":
        return MulticlassImageMixer
    elif task == "multilabel":
        return MultilabelImageMixer
    else:
        raise ValueError(f"Unknown task: {task}")


def select_text_encoder(task, train_dataloader, val_dataloader):
    pass


selectors = {
    'images': select_image_encoder,
    'texts': select_text_encoder,
}
