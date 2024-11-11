from collections import Counter

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader

from auto_mixer.selector import select_encoder_for, select_fusion_strategy


def find_architecture(datamodule: pl.LightningDataModule):
    sampled_train_dataloader = sample_data(datamodule)
    val_dataloader = datamodule.val_dataloader()
    sampled_train_dataloader.target_length = datamodule.target_length
    encoders = select_encoders(sampled_train_dataloader, val_dataloader)
    best_model = select_fusion_strategy(encoders, sampled_train_dataloader, val_dataloader)
    return best_model


def sample_data(dataloader, max_attempts=100, tolerance=0.05):
    generator = torch.Generator().manual_seed(42)
    sample_size = 1000
    # sample_size = calculate_sample_size(len(dataloader.train_dataloader().dataset), 1.96, 0.5, 0.05)
    train_dataloader = dataloader.train_dataloader()
    train_length = len(train_dataloader.dataset)

    original_labels = [train_dataloader.dataset[i]['labels'] for i in range(train_length)]
    original_distribution = Counter(original_labels)
    original_distribution_normalized = {k: v / train_length for k, v in original_distribution.items()}

    for attempt in range(max_attempts):
        # Sample a subset of the dataset
        sub_train_set = Subset(
            train_dataloader.dataset,
            torch.randperm(train_length, generator=generator)[:sample_size]
        )
        sampled_labels = [sub_train_set[i]['labels'] for i in range(len(sub_train_set))]
        sampled_distribution = Counter(sampled_labels)
        sampled_distribution_normalized = {k: v / sample_size for k, v in sampled_distribution.items()}

        # Compare the distributions
        distribution_matches = True
        for label in original_distribution_normalized:
            original_ratio = original_distribution_normalized[label]
            sampled_ratio = sampled_distribution_normalized.get(label, 0)
            print(f"Original ratio for label {label}: {original_ratio}")
            print(f"Sampled ratio for label {label}: {sampled_ratio}")
            if abs(original_ratio - sampled_ratio) > tolerance:
                distribution_matches = False
                break

        if distribution_matches:
            print(f"Sampled dataset matches the original distribution within tolerance on attempt {attempt + 1}")
            sub_train_dataloader = DataLoader(sub_train_set, batch_size=train_dataloader.batch_size,
                                              num_workers=train_dataloader.num_workers, shuffle=True, persistent_workers=True)
            print(f"Sampled dataset size: {len(sub_train_dataloader.dataset)}")
            return sub_train_dataloader
        else:
            print(f"Attempt {attempt + 1}: Distribution did not match, resampling...")

    print("Failed to sample a dataset with a similar distribution after max attempts")
    sub_train_dataloader = DataLoader(sub_train_set, batch_size=train_dataloader.batch_size,
                                      num_workers=train_dataloader.num_workers, shuffle=True, persistent_workers=True)
    return sub_train_dataloader

# Usage example (assuming you have a dataloader object):
# sampled_dataloader = sample_data(dataloader)



def calculate_sample_size(N, z, p_hat, epsilon):
    n = (z ** 2 * p_hat * (1 - p_hat)) / (epsilon ** 2)
    N_prime = n / (1 + ((z ** 2 * p_hat * (1 - p_hat)) / (epsilon ** 2 * N)))
    return N_prime


def select_encoders(sampled_train_dataloader, val_dataloader):
    sample = sampled_train_dataloader.dataset[0]
    modalities = list(sample.keys())
    modalities.remove('labels')
    try:
        task = "multiclass" if len(sample['labels']) == 1 else "multilabel"
    except TypeError:
        task = "multiclass"
    print("\nStarting encoder selection...\n")
    encoders = dict()
    for modality in modalities:
        print(f"Selecting encoder for {modality} modality...")
        encoders[modality] = select_encoder_for(modality, task, sampled_train_dataloader, val_dataloader)

    return encoders
