import pytest
import torch

from auto_mixer.modules.monarch_mixer import MonarchMatrix, MonarchMixerLayer, MonarchMixer


@pytest.fixture
def setup_data():
    sqrt_n = 2
    sqrt_d = 2
    x = torch.randn((2, sqrt_n ** 2, sqrt_d ** 2))
    return sqrt_n, sqrt_d, x


def test_monarch_matrix(setup_data):
    sqrt_n, _, x = setup_data
    model = MonarchMatrix(sqrt_n)
    output = model(x)
    assert output.shape == x.shape


def test_monarch_mixer_layer(setup_data):
    sqrt_n, sqrt_d, x = setup_data
    model = MonarchMixerLayer(sqrt_n, sqrt_d)
    output = model(x)
    assert output.shape == x.shape


@pytest.fixture
def mixer():
    return MonarchMixer(patch_size=16, hidden_dim=64, num_mixers=2)


def test_forward_pass_with_valid_input(mixer):
    input_tensor = torch.randn(10, 16, 64)
    output = mixer(input_tensor)
    assert output.shape == (10, 16, 64)


def test_forward_pass_with_input_not_perfect_square(mixer):
    input_tensor = torch.randn(10, 18, 69)
    output = mixer(input_tensor)
    assert output.shape == (10, 16, 64)
