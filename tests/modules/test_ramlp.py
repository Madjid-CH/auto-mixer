import torch

from auto_mixer.modules.ramlp import Downsampling, SepConv, MLP, RaBlock, RaMLP


def test_downsampling():
    downsampling = Downsampling(3, 64, 7)
    input_tensor = torch.randn(3, 3, 224, 224)
    output = downsampling(input_tensor)
    assert output.shape == (3, 218, 218, 64)


def test_sep_conv():
    sc = SepConv(dim=64, expansion_ratio=2, bias=False, kernel_size=8, head_dim=1)
    input_tensor = torch.randn(1, 224, 224, 64)
    output = sc(input_tensor)
    assert output.shape == (1, 224, 224, 64)


def test_mlp_forward_with_valid_input():
    mlp = MLP(dim=64, mlp_ratio=4)
    input_tensor = torch.randn(1, 64)
    output = mlp(input_tensor)
    assert output.shape == (1, 64)


def test_mlp_forward_with_zero_input():
    mlp = MLP(dim=64, mlp_ratio=4)
    input_tensor = torch.zeros(1, 64)
    output = mlp(input_tensor)
    assert torch.all(output == 0)


def test_ra_block_forward():
    ra_block = RaBlock(dim=64, mlp_ratio=4, expansion_ratio=2, kernel_size=8, head_dim=1)
    input_tensor = torch.randn(1, 64, 64, 64)
    output = ra_block(input_tensor)
    assert output.shape == (1, 64, 64, 64)


def test_ra_mlp_forward():
    ra_mlp = RaMLP()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = ra_mlp(input_tensor)
    assert output.shape == (1, 7, 512)
