import pytest
import torch
from omegaconf import DictConfig

from m2_mixer.usecases.sst.models import NUM_CLASSES, SSTMixer


def hypermixer_config():
    cfg = {
        "dropout": 0.1,
        "modalities": {
            "classification": {
                "input_dim": 66,
                "hidden_dim": 512,
                "output_dim": NUM_CLASSES
            },
            "block_type": "TextHyperMixer",
            "text": {
                "hidden_dim": 768,
                "patch_size": 66,
                "channel_dim": 1900,
                "num_mixers": 7,
                "num_heads": 2
            }
        }
    }
    return DictConfig(cfg)


def monarchmixer_config():
    cfg = {
        'dropout': 0.1,
        'modalities': {
            'classification': {
                'input_dim': 64,
                'hidden_dim': 512,
                'output_dim': NUM_CLASSES
            },
            'block_type': 'MonarchMixer',
            'text': {
                'hidden_dim': 768,
                'patch_size': 66,
                'channel_dim': 4096,
                'num_mixers': 160
            }
        }
    }
    return DictConfig(cfg)


def mlpmixer_config():
    cfg = {
        "dropout": 0.1,
        "modalities": {
            "classification": {
                "input_dim": 66,
                "hidden_dim": 512,
                "output_dim": NUM_CLASSES
            },
            "block_type": "TextMLPMixer",
            "text": {
                "hidden_dim": 768,
                "patch_size": 66,
                "token_dim": 256,
                "channel_dim": 2048,
                "num_mixers": 72
            }
        }
    }
    return DictConfig(cfg)


@pytest.fixture(params=[hypermixer_config(), monarchmixer_config(), mlpmixer_config()],
                ids=['hypermixer', 'monarchmixer', 'mlpmixer'])
def model(request, optimizer_cfg):
    config = request.param
    return SSTMixer(model_cfg=config, optimizer_cfg=optimizer_cfg)


def test_small_model_size(model):
    import numpy as np
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    assert 20 < parameters < 22


@pytest.fixture
def batch():
    return torch.randn(10, 66, 768), torch.randint(0, NUM_CLASSES, (10,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_shared_step_returns_correct_keys(model, batch):
    output = model.shared_step(batch)
    assert set(output.keys()) == {'preds', 'labels', 'loss', 'logits'}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_get_logits_returns_correct_shape(model, batch):
    logits = model.get_logits(batch[0])
    assert logits.shape == (10, NUM_CLASSES)
