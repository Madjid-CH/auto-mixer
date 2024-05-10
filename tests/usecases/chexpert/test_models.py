import pytest
import torch
from omegaconf import DictConfig

from m2_mixer.usecases.chexpert.models import NUM_LABELS, ChexpertMixer


def hypermixer_config():
    cfg = {
        "dropout": 0.3,
        "modalities": {
            "classification": {
                "input_dim": 512,
                "hidden_dim": 512,
                "output_dim": NUM_LABELS
            },
            "block_type": "HyperMixer",
            "image": {
                "in_channels": 3,
                "hidden_dim": 512,
                "patch_size": 16,
                "image_size": [224, 224],
                "channel_dim": 2048,
                "num_mixers": 9,
                "num_heads": 2
            }
        }
    }
    return DictConfig(cfg)


def ramlp_config():
    cfg = {
        'dropout': 0.3,
        'modalities': {
            'classification': {
                'input_dim': 512,
                'hidden_dim': 512,
                'output_dim': NUM_LABELS
            },
            "block_type": "RaMLP",
            "image": {
                "in_channels": 3,
                "patch_size": 16,
                "depths": [3, 3, 10, 3],
                "dims": [64, 128, 256, 512],
                "mlp_ratio": [4, 4, 3, 3],
                "expansion_ratio": [3, 3, 2, 2],
                "kernel_size": [8, 4, 2, 1],
                "head_dims": [1, 4, 16, 64]
            }
        },
    }
    return DictConfig(cfg)


def mlpmixer_config():
    cfg = {
        "dropout": 0.5,
        "modalities": {
            "classification": {
                "input_dim": 512,
                "hidden_dim": 512,
                "output_dim": NUM_LABELS
            },
            "block_type": "MLPMixer",
            "image": {
                "in_channels": 3,
                "hidden_dim": 512,
                "patch_size": 16,
                "image_size": [224, 224],
                "token_dim": 256,
                "channel_dim": 2048,
                "num_mixers": 9
            }
        }
    }
    return DictConfig(cfg)


def stripmlp_config():
    cfg = {
        "dropout": 0.3,
        "modalities": {
            "classification": {
                "input_dim": 640,
                "hidden_dim": 512,
                "output_dim": NUM_LABELS
            },
            "block_type": "StripMLP",
            "image": {
                "in_channels": 3,
                "embeddings_dim": 80,
                "patch_size": 16,
                "image_size": [224, 224],
                "layers": [2, 6, 9, 2]
            }
        }
    }
    return DictConfig(cfg)


@pytest.fixture(params=[hypermixer_config(), ramlp_config(), mlpmixer_config(), stripmlp_config()],
                ids=['hypermixer', 'ramlp', 'mlpmixer', 'stripmlp'])
def model(request, optimizer_cfg):
    config = request.param
    return ChexpertMixer(model_cfg=config, optimizer_cfg=optimizer_cfg)


def test_small_model_size(model):
    import numpy as np
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    assert 20 < parameters < 22


@pytest.fixture
def batch():
    return torch.randn(10, 3, 224, 224), torch.randint(0, 2, (10, NUM_LABELS))


def test_shared_step_returns_correct_keys(model, batch):
    output = model.shared_step(batch)
    assert set(output.keys()) == {'preds', 'labels', 'loss', 'logits'}


def test_get_logits_returns_correct_shape(model, batch):
    logits = model.get_logits(batch[0])
    assert logits.shape == (10, NUM_LABELS)
