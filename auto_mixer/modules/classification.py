import torch
import torch.nn as nn


class StandardClassifier(nn.Module):
    def __init__(self, input_shape: tuple, num_classes: int, **_kwargs):
        super(StandardClassifier, self).__init__()
        self.classifier = nn.Linear(input_shape[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.reshape(inputs.shape[0], -1, inputs.shape[-1]).mean(dim=1))
