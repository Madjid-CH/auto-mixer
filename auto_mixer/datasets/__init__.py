from datasets.mm_imdb.get_processed_mmimdb import *
from datasets.mm_imdb.get_processed_mmimdb import *
from m2_mixer.usecases.text_db.dataset import *
from m2_mixer.usecases.sst.dataset import *


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    return getattr(sys.modules[__name__], data_type)
