import sys

import pytorch_lightning as pl

from .avmnist import *
from .avmnist_post import *
from .mmimdb_gmlp import *
from .mmimdb import *
from .multioff import *
from .mmhs150 import *
from .mimic import *
from m2_mixer.usecases.text_db.models import *
from m2_mixer.usecases.imagenet.models import *
from m2_mixer.usecases.chexpert.models import *
from m2_mixer.usecases.sst.models import *


def get_model(model_type: str) -> type[pl.LightningModule]:
    return getattr(sys.modules[__name__], model_type)
