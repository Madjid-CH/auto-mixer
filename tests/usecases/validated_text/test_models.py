import pytest
import torch
from omegaconf import DictConfig

from m2_mixer.usecases.text_db.models import TextEmbeddingMixer, NUM_CLASSES


def hypermixer_config():
    cfg = {
        "dropout": 0.3,
        "modalities": {
            "classification": {
                "input_dim": 298,
                "hidden_dim": 512,
                "output_dim": NUM_CLASSES,
            },
            "block_type": "TextHyperMixer",
            "text": {
                "hidden_dim": 768,
                "patch_size": 298,
                "channel_dim": 128,
                "num_mixers": 4,
            }
        }
    }
    return DictConfig(cfg)


def monarchmixer_config():
    cfg = {
        'dropout': 0.3,
        'modalities': {
            'classification': {
                'input_dim': 289,
                'hidden_dim': 512,
                'output_dim': NUM_CLASSES
            },
            'block_type': 'MonarchMixer',
            'text': {
                'hidden_dim': 768,
                'patch_size': 298,
                'channel_dim': 128,
                'num_mixers': 4
            }
        }
    }
    return DictConfig(cfg)


def tlp_config():
    cfg = {
        'dropout': 0.3,
        'type': 'TextEmbeddingMixer',
        'modalities': {
            'classification': {
                'input_dim': 298,
                'hidden_dim': 512,
                'output_dim': NUM_CLASSES
            },
            'block_type': 'TwoLayeredPerceptron',
            'text': {
                'input_dim': 768,
                'hidden_dim': 298,
                'output_dim': 512
            }
        }
    }
    return DictConfig(cfg)


def debug_model_config():
    cfg = {
        'dropout': 0.3,
        'type': 'TextEmbeddingMixer',
        'modalities': {
            'classification': {
                'input_dim': 298,
                'hidden_dim': 32,
                'output_dim': NUM_CLASSES
            },
            'block_type': 'TwoLayeredPerceptron',
            'text': {
                'input_dim': 768,
                'hidden_dim': 32,
                'output_dim': 32
            }
        }
    }
    return DictConfig(cfg)


@pytest.fixture(params=[hypermixer_config(), monarchmixer_config(), tlp_config(), debug_model_config()],
                ids=['hypermixer', 'monarchmixer', 'tlp', 'debug model'])
def model(request, optimizer_cfg):
    config = request.param
    return TextEmbeddingMixer(model_cfg=config, optimizer_cfg=optimizer_cfg)


@pytest.fixture
def batch():
    return torch.randn(10, 298, 768), torch.randint(0, NUM_CLASSES, (10,))


def test_shared_step_returns_correct_keys(model, batch):
    output = model.shared_step(batch)
    assert set(output.keys()) == {'preds', 'labels', 'loss', 'logits'}


def test_get_logits_returns_correct_shape(model, batch):
    logits = model.get_logits(batch[0])
    assert logits.shape == (10, NUM_CLASSES)
