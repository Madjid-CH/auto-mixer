import pytest
from omegaconf import DictConfig


@pytest.fixture
def optimizer_cfg():
    cfg = {
        "lr": 5e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8
    }
    return DictConfig(cfg)
