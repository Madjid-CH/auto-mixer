import torch
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl

from selector import select_encoder_for


def train(dataloader: pl.LightningDataModule):
    sampled_train_dataloader = sample_data(dataloader)
    val_dataloader = dataloader.val_dataloader()
    encoders = select_encoders(sampled_train_dataloader, val_dataloader)


def sample_data(dataloader):
    generator = torch.Generator().manual_seed(42)
    sample_size = 1000
    train_dataloader = dataloader.train_dataloader()
    train_length = len(train_dataloader.dataset)
    sub_train_set = Subset(
        train_dataloader.dataset,
        torch.randperm(train_length, generator=generator)[:sample_size]
    )
    sub_train_dataloader = DataLoader(sub_train_set, batch_size=train_dataloader.batch_size,
                                      num_workers=train_dataloader.num_workers, shuffle=True, persistent_workers=True)
    return sub_train_dataloader


def select_encoders(sampled_train_dataloader, val_dataloader):
    sample = sampled_train_dataloader.dataset[0]
    modalities = list(sample.keys())
    modalities.remove('labels')
    task = "multiclass" if len(sample['labels'].shape) == 1 else "multilabel"
    encoders = dict()
    for modality in modalities:
        encoders[modality] = select_encoder_for(modality, task, sampled_train_dataloader, val_dataloader)

    return encoders
