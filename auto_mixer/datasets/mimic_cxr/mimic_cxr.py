import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from auto_mixer.utils.utils import get_env_var


class MIMICCXR(Dataset):
    def __init__(self, stage='train', max_seq_len=301, transform=None, **_kwargs):
        super().__init__()
        self.root_dir = get_env_var("MIMIC_CXR_DIR")
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.images, self.embeddings, self.labels = self._setup_data()
        self.transform = transform

    def _setup_data(self):
        if self.stage == 'train':
            df = pd.read_pickle(f"{self.root_dir}/mimic_cxr_train.pkl")
        elif self.stage == 'dev':
            df = pd.read_pickle(f"{self.root_dir}/mimic_cxr_val.pkl")
        else:
            df = pd.read_pickle(f"{self.root_dir}/mimic_cxr_test.pkl")

        images = df['jpg_path'].values
        embeddings = df['embeddings'].values
        labels = df['labels'].values
        labels = torch.tensor([l for l in labels])

        return images, embeddings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        image_path = f"{self.root_dir}/{self.images[idx]}"
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        embeddings = self.embeddings[idx]
        embeddings = self._pad_embeddings(embeddings)
        sample = {'images': image, 'texts': embeddings.astype("float"), 'labels': label}

        if self.transform:
            sample = self._apply_transformation(sample)

        return sample

    def _pad_embeddings(self, embeddings):
        padded = np.zeros((self.max_seq_len, 768))
        padded[:embeddings.shape[0], :] = embeddings
        return padded

    def _apply_transformation(self, sample):
        for m in self.transform:
            if m == 'images':
                sample[m] = self.transform[m](sample[m])
            elif m == 'multimodal':
                sample = self.transform[m](sample)
        return sample


class MIMICCXRDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, max_seq_len: int = 301,
                 **_kwargs):
        super().__init__()
        self.padded_features = None
        self.train_set = None
        self.eval_set = None
        self.test_set = None
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def target_length(self):
        return 14

    def setup(self, **_stage):
        train_transforms = dict(images=T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=(-20, 20), translate=(0.09, 0.1), scale=(0.95, 1.05)),
            T.RandomApply(
                [T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4))], p=0.5
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.0))], p=0.5
            ),

            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]),
        )

        val_test_transforms = dict(images=T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

        self.train_set = MIMICCXR(stage='train', max_seq_len=self.max_seq_len, transform=train_transforms)
        self.eval_set = MIMICCXR(stage='dev', max_seq_len=self.max_seq_len, transform=val_test_transforms)
        self.test_set = MIMICCXR(stage='test', max_seq_len=self.max_seq_len, transform=val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)
