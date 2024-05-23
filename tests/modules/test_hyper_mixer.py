import pytest
import torch

from auto_mixer.modules.hyper_mixer import HyperMixing, HyperMixerBlock, HyperMixer, TextHyperMixer


@pytest.fixture
def mixer():
    return HyperMixing(input_output_dim=128,
                       hypernet_size=512,
                       tied=False,
                       num_heads=2,
                       max_length=300)


def test_hypermixing_forward(mixer):
    queries = torch.randn((64, 50, 128))
    keys = torch.randn((64, 25, 128))
    values = torch.randn((64, 25, 128))
    out = mixer(queries, keys, values)
    assert out.size() == queries.size()


def test_hypermixing_invalid_input():
    with pytest.raises(ValueError):
        HyperMixing(input_output_dim=512, hypernet_size=2048, num_heads=2)(
            torch.rand([8, 60, 512]),
            torch.rand([8, 70, 512]),
            torch.rand([8, 60, 512])
        )


@pytest.fixture
def hyper_mixer_block():
    hidden_dim = 512
    num_patch = 16
    channel_dim = 2048
    num_heads = 2
    dropout = 0.1
    return HyperMixerBlock(hidden_dim, num_patch, channel_dim, num_heads, dropout)


def test_forward(hyper_mixer_block):
    num_patch = 16
    hidden_dim = 512
    x = torch.rand(1, num_patch, hidden_dim, dtype=torch.float32)
    output = hyper_mixer_block(x)
    assert output.shape == (1, num_patch, hidden_dim)


def test_hyper_mixer():
    in_channels = 3
    hidden_dim = 64
    patch_size = 4
    image_size = (32, 32)
    num_mixers = 2
    channel_dim = 32
    num_heads = 2
    dropout = 0.1

    hyper_mixer = HyperMixer(in_channels, hidden_dim, patch_size,
                             image_size, num_mixers, channel_dim, num_heads,
                             dropout)
    batch_size = 10
    images = torch.rand(batch_size, in_channels, image_size[0], image_size[1])
    output = hyper_mixer(images)
    assert output.shape == (batch_size, hyper_mixer.num_patch, hidden_dim), "Output shape is incorrect"


def test_text_hyper_mixer_forward():
    model = TextHyperMixer(hidden_dim=768, num_mixers=2, num_patch=512, channel_dim=128)
    input_tensor = torch.rand(10, 512, 768)
    output = model(input_tensor)
    assert output.shape == (10, 512, 768)


