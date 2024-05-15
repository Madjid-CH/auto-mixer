import sys

import pytorch_lightning as pl

from .image_models import MulticlassImageMixer, MultilabelImageMixer  # noqa: F401
from .text_models import MulticlassTextMixer, MultilabelTextMixer  # noqa: F401


def get_model(model_type: str) -> type[pl.LightningModule]:
    return getattr(sys.modules[__name__], model_type)
