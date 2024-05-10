from pathlib import Path

import pytest

from m2_mixer.usecases.chexpert.dataset import ChexpertDataset, ChexpertDataModule
from m2_mixer.utils.utils import get_env_var


def dataset_exists():
    if get_env_var('CHEXPERT_DIR') is None:
        return False
    return Path(get_env_var('CHEXPERT_DIR')).exists()


@pytest.mark.skipif(not dataset_exists(), reason="Chexpert Dataset not found.")
@pytest.mark.parametrize("stage", ['train', 'val', 'test'])
class TestChexpertDataset:
    def test_get_item(self, stage):
        dataset = ChexpertDataset(stage=stage)
        image, labels = dataset[0]
        assert image.shape == (3, 224, 224)
        assert len(labels) == 14


@pytest.mark.skipif(not dataset_exists(), reason="Chexpert Dataset not found.")
class TestChexpertDataModule:
    @pytest.mark.parametrize("proportion, length", [(0.1, 22_319), (0.3, 66_957), (1.0, 223_191)],
                             ids=["10%", "30%", "100%"])
    def test_train_set_proportions(self, proportion, length):
        d = ChexpertDataModule(batch_size=1, num_workers=1, proportion=proportion)
        d.setup('fit')
        assert len(d.train_set) == length

    @pytest.mark.parametrize("proportion", [0.2, 0.6, 0.8], ids=["10%", "30%", "100%"])
    def test_val_set_and_test_set_does_not_change_with_proportion(self, proportion):
        d = ChexpertDataModule(batch_size=1, num_workers=1, proportion=proportion)
        d.setup('fit')
        d.setup('test')
        assert len(d.val_set) == 223
        assert len(d.test_set) == 234
