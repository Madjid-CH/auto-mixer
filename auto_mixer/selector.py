import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from auto_mixer.models import MulticlassImageMixer, MultilabelImageMixer, MulticlassTextMixer, MultilabelTextMixer
from auto_mixer.models.multimodal_models import MultiClassMultiLoss, MultiLabelMultiLoss


def select_encoder_for(modality, task, train_dataloader, val_dataloader):
    selector = selectors[modality]
    encoder = selector(task, train_dataloader, val_dataloader)
    return encoder


def select_image_encoder(task, train_dataloader, val_dataloader):
    cfgs_path = "auto_mixer/cfg/image_models"
    modules_configs_files = os.listdir(cfgs_path)
    cfgs = [OmegaConf.load(f"{cfgs_path}/{cfg_file}") for cfg_file in modules_configs_files]
    Mixer = get_image_model_for(task)
    image_size = train_dataloader.dataset[0]['images'].shape[1:]
    train_cfg = OmegaConf.load("auto_mixer/cfg/micro_train.yml")
    models = {
        cfg.block_type: Mixer(image_size=image_size,
                              target_length=train_dataloader.target_length,
                              model_cfg=cfg,
                              optimizer_cfg=train_cfg.optimizer
                              ) for cfg in cfgs
    }
    return benchmark(models, train_cfg, train_dataloader, val_dataloader)


def benchmark(models, train_cfg, train_dataloader, val_dataloader):
    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        log_every_n_steps=train_cfg.log_interval_steps,
        max_epochs=train_cfg.epochs
    )
    for block_type, model in models.items():
        print(f"Benchmarking {block_type} model...")
        trainer.fit(model, train_dataloader, val_dataloader)
        results = trainer.test(model, val_dataloader)[0]
        model.results = results
    block_type, best_model = max(models.items(), key=lambda x: x[1].results['test_accuracy'])
    return block_type, best_model.backbone


def get_image_model_for(task):
    if task == "multiclass":
        return MulticlassImageMixer
    elif task == "multilabel":
        return MultilabelImageMixer
    else:
        raise ValueError(f"Unknown task: {task}")


def select_text_encoder(task, train_dataloader, val_dataloader):
    cfgs_path = "auto_mixer/cfg/text_models"
    modules_configs_files = os.listdir(cfgs_path)
    cfgs = [OmegaConf.load(f"{cfgs_path}/{cfg_file}") for cfg_file in modules_configs_files]
    train_cfg = OmegaConf.load("auto_mixer/cfg/micro_train.yml")
    Mixer = get_text_model_for(task)
    patch_size, hidden_dim = train_dataloader.dataset[0]['texts'].shape
    models = {
        cfg.block_type: Mixer(
            hidden_dim=hidden_dim, patch_size=patch_size,
            target_length=train_dataloader.target_length, model_cfg=cfg,
            optimizer_cfg=train_cfg.optimizer
        ) for cfg in cfgs
    }
    return benchmark(models, train_cfg, train_dataloader, val_dataloader)


def get_text_model_for(task):
    if task == "multiclass":
        return MulticlassTextMixer
    elif task == "multilabel":
        return MultilabelTextMixer
    else:
        raise ValueError(f"Unknown task: {task}")


selectors = {
    'images': select_image_encoder,
    'texts': select_text_encoder,
}


def select_fusion_strategy(encoders, train_dataloader, val_dataloader):
    modules_configs_files = os.listdir("auto_mixer/cfg/fusion_models")
    cfgs = [OmegaConf.load(cfg_file) for cfg_file in modules_configs_files]
    train_cfg = OmegaConf.load("auto_mixer/cfg/micro_train.yml")
    sample = train_dataloader.dataset[0]
    task = "multiclass" if len(sample['labels']) == 1 else "multilabel"
    Mixer = get_multimodal_model_for(task)
    models = {
        cfg.modalities.fusion_function: Mixer(
            encoders=encoders,
            target_length=train_dataloader.target_length,
            model_cfg=cfg,
            optimizer_cfg=train_cfg.optimizer
        ) for cfg in cfgs
    }

    return benchmark_fusion(models, train_cfg, train_dataloader, val_dataloader)


def get_multimodal_model_for(task):
    if task == "multiclass":
        return MultiClassMultiLoss
    elif task == "multilabel":
        return MultiLabelMultiLoss
    else:
        raise ValueError(f"Unknown task: {task}")


def benchmark_fusion(models, train_cfg, train_dataloader, val_dataloader):
    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        log_every_n_steps=train_cfg.log_interval_steps,
        max_epochs=train_cfg.epochs
    )
    for fusion_function, model in models.items():
        print(f"Benchmarking {fusion_function}...")
        trainer.fit(model, train_dataloader, val_dataloader)
        results = trainer.test(model, val_dataloader)[0]
        model.results = results
    fusion_function, best_model = max(models.items(), key=lambda x: x[1].results['test_accuracy'])
    return fusion_function, best_model
