import argparse
import importlib
import os

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
import yaml_include
from auto_mixer import datasets
import models
import wandb
from auto_mixer.utils.utils import deep_update, todict

torch.set_float32_matmul_precision('high')

yaml.SafeLoader.add_constructor('!include', yaml_include.Constructor(base_dir='.'))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--ckpt', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-pr', '--project', type=str, default='mmimdb-experiments')
    parser.add_argument('--disable-wandb', action='store_true', default=False)
    parser.add_argument('--save-ckpt', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    return args, unknown


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


def upload_checkpoint_to_wandb(trainer):
    os.environ["WANDB_DISABLE_SYMLINKS"] = "1"
    checkpoint_dir = trainer.checkpoint_callback.dirpath
    checkpoint_files = os.listdir(checkpoint_dir)
    for file in checkpoint_files:
        wandb.save(os.path.join(checkpoint_dir, file))


def main():
    args, unknown = parse_args()
    cfg = OmegaConf.load(args.cfg)
    train_cfg = cfg.train
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    pl.seed_everything(train_cfg.seed)
    unknown = [u.replace('--', '') for u in unknown]
    ucfg = OmegaConf.from_cli(unknown)
    if 'model' in ucfg:
        model_cfg = deep_update(model_cfg, ucfg.model)
    if 'train' in ucfg:
        train_cfg = deep_update(train_cfg, ucfg.train)
    if 'dataset' in ucfg:
        dataset_cfg = deep_update(dataset_cfg, ucfg.dataset)
    if args.disable_wandb:
        wandb.init(project=args.project, name=args.name, config=todict(cfg), mode='disabled')
    else:
        wandb.init(project=args.project, name=args.name, config=todict(cfg))
    model = models.get_model(model_cfg.type)
    if args.ckpt:
        train_module = model.load_from_checkpoint(args.ckpt,
                                                  optimizer_cfg=train_cfg.optimizer,
                                                  model_cfg=model_cfg)
    else:
        train_module = model(model_cfg, train_cfg.optimizer)
    wandb.watch(train_module)
    data_module = datasets.get_data_module(dataset_cfg.type)
    if dataset_cfg.params.num_workers == -1:
        dataset_cfg.params.num_workers = os.cpu_count()
    data_module = data_module(**dataset_cfg.params)
    callbacks = build_callbacks(train_cfg.callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        devices=torch.cuda.device_count(),
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs
    )
    wandb.config.update({"run_version": trainer.logger.version})
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
        trainer.test(train_module, data_module, ckpt_path='best')
        if args.save_ckpt:
            upload_checkpoint_to_wandb(trainer)
    if args.mode == 'test':
        trainer.test(train_module, data_module)


if __name__ == '__main__':
    main()
