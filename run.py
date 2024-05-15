import importlib

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf

from auto_mixer.datasets.mm_imdb.mmimdb import MMIMDBDataModule
from auto_mixer.runner import find_architecture


def main():
    wandb.init(project='auto-mixer', name='test', mode='disabled')
    data = MMIMDBDataModule(batch_size=32, num_workers=4, max_seq_len=512,
                            dataset_cls_name="MMIMDBDatasetWithEmbeddings")
    data.setup()
    fusion_function, best_model = find_architecture(data)
    cfg = OmegaConf.load("auto_mixer/cfg/train.yml")
    train_cfg = cfg.train
    callbacks = build_callbacks(train_cfg.callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        devices=torch.cuda.device_count(),
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, "pipeline"),
        max_epochs=train_cfg.epochs
    )

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
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
