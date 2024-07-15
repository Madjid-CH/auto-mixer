"""
This module implements the Monarch Mixer, a sub-quadratic architecture for language and vision tasks.

The Monarch Mixer uses Monarch matrices to mix information along both the sequence and model dimension axes.
 It introduces the M2 layer, which consists of two mixing operations: one along the sequence axis,
  and one along the embedding axis.
  The sequence mixer can implement convolutions or gated convolutions using Monarch matrices,
  and the dimension mixer can replace the dense matrices in MLPs with Monarch matrices.

source: https://arxiv.org/pdf/2310.12109.pdf
"""

import torch
from einops import rearrange
from torch import nn


def blockdiag_matmul(x, w):
    """
    Perform a block diagonal matrix multiplication between the input tensor x and the weight matrix w.
    """
    return torch.einsum(
        "bnm,...bm->...bn",
        w,
        x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    ).reshape(*x.shape)


class MonarchMatrix(nn.Module):
    def __init__(self, sqrt_n: int):
        super().__init__()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.R = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))

    def forward(self, x):
        x = rearrange(x, "... (m n)-> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n)-> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)
        return rearrange(x, "... (m n)-> ... (n m)", n=self.sqrt_n)


class MonarchMixerLayer(nn.Module):
    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n)
        self.m2 = MonarchMatrix(sqrt_n)
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

        self.n_kernel = nn.Parameter(torch.randn(sqrt_d ** 2, sqrt_n ** 2))
        self.d_kernel = nn.Parameter(torch.randn(1, sqrt_d ** 2))
        self.layer_norm = nn.LayerNorm(sqrt_d ** 2)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MonarchMixerLayer, first it mixes the sequence and then the features of the input tensor.
        In addition, it applies a skip connection.
        :param x: Input tensor of shape (b, n, d)
        """
        x_tilde = self.m2(self.n_kernel * self.m1(x.transpose(-1, -2))).transpose(-1, -2)
        y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde)))
        return self.layer_norm(y + x_tilde)


class MonarchMixer(nn.Module):
    def __init__(self, patch_size: int, hidden_dim: int, num_mixers: int, **_kwargs):
        super().__init__()
        self.sqrt_n = int(patch_size ** 0.5)
        self.sqrt_d = int(hidden_dim ** 0.5)
        self.hidden_dim = self.sqrt_d ** 2
        self.num_patch = self.sqrt_n ** 2
        self.layers = nn.ModuleList([MonarchMixerLayer(self.sqrt_n, self.sqrt_d) for _ in range(num_mixers)])

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MonarchMixer, applies the layers in sequence.
        :param x: Input tensor of shape (b, n, d)
        """
        x = self._trim_size_to_perfect_square(x.double())
        for layer in self.layers:
            x = layer(x)
        return x

    def _trim_size_to_perfect_square(self, x):
        """
        trim the input size to perfect square.
        """
        n = self.sqrt_n ** 2
        d = self.sqrt_d ** 2
        return x[:, :n, :d]
