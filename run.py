import importlib

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from auto_mixer.datasets.mimic_cxr.mimic_cxr import MIMICCXRDataModule
from auto_mixer.runner import find_architecture


def main():
    data = MIMICCXRDataModule(batch_size=64, num_workers=4)
    data.setup()
    fusion_function, best_model = find_architecture(data)
    cfg = OmegaConf.load("auto_mixer/cfg/train.yml")
    wandb_logger = WandbLogger(project='auto-mixer', name='mimic_cxr')
    wandb_logger.experiment.config.update(cfg)
    callbacks = build_callbacks(cfg.callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        devices=torch.cuda.device_count(),
        log_every_n_steps=cfg.log_interval_steps,
        logger=wandb_logger,
        max_epochs=cfg.epochs
    )

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    print("\nStarting full training...\n")
    print(best_model)
    trainer.fit(best_model, train_dataloader, val_dataloader)
    results = trainer.test(best_model, val_dataloader)[0]
    print(results)


def build_callbacks(callbacks_cfg):
    callbacks = []
    for cb in callbacks_cfg:
        callbacks.append(build_callback(cb))
    return callbacks


def build_callback(cfg):
    module_path, class_name = cfg.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    callback = cls(**cfg.init_args)
    return callback


if __name__ == '__main__':
    main()
