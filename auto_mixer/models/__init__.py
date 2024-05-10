import sys

import pytorch_lightning as pl

from .avmnist import *
from .avmnist_post import *
from .mimic import *
from .mmhs150 import *
from .mmimdb import *
from .mmimdb_gmlp import *
from .multioff import *


def get_model(model_type: str) -> type[pl.LightningModule]:
    return getattr(sys.modules[__name__], model_type)
