from timm.models.metaformer import SquaredReLU, StarReLU
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim=None, dropout=0., **kwargs):
        super().__init__()

        self.module_list = nn.ModuleList()
        self.output_dim = output_dim

        for i in range(num_blocks):
            if i == 0:
                self.module_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.module_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.module_list.append(nn.ReLU())
            self.module_list.append(nn.Dropout(dropout))

        if output_dim is not None:
            self.module_list.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x


class WideMLP(nn.Module):
    """
     MLP as used in Vision Transformer, MLP-Mixer and related networks
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        mid_dim = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, mid_dim, bias=bias)
        self.act = StarReLU()
        self.fc2 = nn.Linear(mid_dim, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MLPClassificationHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU, norm_layer=nn.LayerNorm,
                 head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class TwoLayeredPerceptron(nn.Module):
    """
    Adapted from: https://github.com/pliang279/MultiBench/blob/main/unimodals/common_models.py#L134
    """

    def __init__(self, input_dim, hidden_dim, output_dim, apply_dropout=False, dropout=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output layer dimension
            apply_dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(TwoLayeredPerceptron, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout = apply_dropout
        self.output_each_layer = output_each_layer
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.activation(output2)]
        return output2
