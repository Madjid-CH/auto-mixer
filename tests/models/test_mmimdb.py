import pytest
import torch
from omegaconf import DictConfig

from models import MMImdbImageMixer


@pytest.fixture
def tlp_config():
    cfg = {
        'type': 'MMImdbImageMixer',
        'dropout': 0.3,
        'modalities': {
            'classification': {
                'input_dim': 128,
                'hidden_dim': 512,
                'output_dim': 23
            },
            'block_type': 'SmallCNN',
            'image': {
                'in_channels': 3,
                'hidden_dim': 128,
                'patch_size': 16,
                'image_size': [160, 256]
            }
        }
    }
    return DictConfig(cfg)


@pytest.fixture
def optimizer_cfg():
    cfg = {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8
    }
    return DictConfig(cfg)


@pytest.fixture()
def model(tlp_config, optimizer_cfg):
    return MMImdbImageMixer(model_cfg=tlp_config, optimizer_cfg=optimizer_cfg)


@pytest.fixture
def batch():
    return {'image': torch.randn(10, 3, 160, 256), "label": torch.randint(0, 2, (10, 23))}


def test_model(model, batch):
    out = model.shared_step(batch)
    assert out['preds'].shape == (10, 23)
