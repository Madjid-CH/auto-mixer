import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader

from auto_mixer.selector import select_encoder_for, select_fusion_strategy


def train(datamodule: pl.LightningDataModule):
    sampled_train_dataloader = sample_data(datamodule)
    val_dataloader = datamodule.val_dataloader()
    sampled_train_dataloader.target_length = datamodule.target_length
    encoders = select_encoders(sampled_train_dataloader, val_dataloader)
    fusion_strategy = select_fusion_strategy(encoders, sampled_train_dataloader, val_dataloader)


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
    task = "multiclass" if len(sample['labels']) == 1 else "multilabel"
    print("\nStarting encoder selection...\n")
    encoders = dict()
    for modality in modalities:
        print(f"Selecting encoder for {modality} modality...")
        encoders[modality] = select_encoder_for(modality, task, sampled_train_dataloader, val_dataloader)

    return encoders
