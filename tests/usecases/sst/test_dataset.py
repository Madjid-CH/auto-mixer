import numpy as np
import pytest

from m2_mixer.usecases.sst.dataset import SSTDataset, SSTDataModule


@pytest.fixture(scope='session')
def embeddings():
    embeddings = np.random.rand(5, 66, 768)
    return embeddings


@pytest.fixture
def mock_load_data(embeddings, monkeypatch):
    def _mock_load_data(*_args, **_kwargs):
        return embeddings, np.random.randint(0, 5, 5)

    monkeypatch.setattr(SSTDataset, '_load_data', _mock_load_data)


class TestDataset:
    @pytest.mark.parametrize("stage", ['train', 'dev', 'test'])
    def test_loads_data_correctly(self, mock_load_data, stage):
        dataset = SSTDataset(stage=stage)
        assert len(dataset) == 5
        assert len(dataset[0]) == 2


class TestTextDataModule:
    def test_loads_datasets_correctly(self, mock_load_data):
        data_module = SSTDataModule(batch_size=32, num_workers=1)
        data_module.setup(stage='fit')
        assert data_module.train_set is not None
        assert data_module.val_set is not None

    def test_loads_test_dataset_correctly(self, mock_load_data):
        data_module = SSTDataModule(batch_size=32, num_workers=4)
        data_module.setup(stage='test')
        assert data_module.test_set is not None
