import sys

from .fusion import *
from .hyper_mixer import *
from .image_module import MultilabelImageMixer, MulticlassImageMixer  # noqa: F401
from .mixer import *
from .mlp import *
from .monarch_mixer import *
from .ramlp import *
from .strip_mlp import *
from .text_module import MultilabelTextMixer, MulticlassTextMixer  # noqa: F401
from .wave_mlp import *


def get_block_by_name(**kwargs):
    this_module = sys.modules[__name__]
    block = getattr(this_module, kwargs['block_type'])
    return block(**kwargs)


def get_fusion_by_name(**kwargs):
    this_module = sys.modules[__name__]
    fusion = getattr(this_module, kwargs['fusion_function'])
    return fusion(**kwargs)


def get_classifier_by_name(**kwargs):
    this_module = sys.modules[__name__]
    classifier = getattr(this_module, kwargs['classifier'])
    return classifier(**kwargs)
