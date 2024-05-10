import numpy as np
import pytest

from m2_mixer.usecases.text_db.dataset import (TextDataset, TextDataModule, MaskedTextDataset, W2VTextDataset,
                                               MaskedW2VTextDataset, AveragedTextDataset, AveragedMaskedTextDataset)


@pytest.fixture(scope='session')
def embeddings():
    embeddings = np.random.rand(5, 298, 300)
    embeddings[:, -50:, :] = 0
    return embeddings

@pytest.fixture
def mock_load_data(embeddings, monkeypatch):
    def _mock_load_data(*_args, **_kwargs):
        return embeddings, np.random.randint(0, 2, 5)

    monkeypatch.setattr(TextDataset, '_load_data', _mock_load_data)


class TestTextDataset:
    @pytest.mark.parametrize("dataset_cls", [TextDataset, MaskedTextDataset, W2VTextDataset, MaskedW2VTextDataset],
                             ids=['TextDataset', 'MaskedTextDataset', 'W2VTextDataset', 'MaskedW2VTextDataset'])
    @pytest.mark.parametrize("stage", ['train', 'val', 'test'])
    def test_loads_data_correctly(self, dataset_cls, mock_load_data, stage):
        dataset = dataset_cls(stage=stage)
        assert len(dataset) == 5
        assert len(dataset[0]) == 2

    @pytest.mark.parametrize("dataset_cls", [AveragedTextDataset, AveragedMaskedTextDataset],
                             ids=['AveragedTextDataset', 'AveragedMaskedTextDataset'])
    @pytest.mark.parametrize("stage", ['train', 'val', 'test'])
    class TestAveragedTextDataset:

        def test_loads_averaged_data_correctly(self, dataset_cls, mock_load_data, stage):
            dataset = dataset_cls(stage=stage)
            assert len(dataset) == 5
            assert len(dataset[0]) == 2
            assert dataset[0][0].shape == (300,)

        def test_dataset_cls_ignores_padded_values(self, dataset_cls, mock_load_data, stage, embeddings):
            dataset = dataset_cls(stage=stage)
            mean_without_pad = dataset[0][0]
            mean_with_pad = embeddings[0].mean(axis=0)
            assert np.all(mean_without_pad > mean_with_pad)


class TestTextDataModule:
    def test_loads_datasets_correctly(self, mock_load_data):
        data_module = TextDataModule(batch_size=32, num_workers=1, dataset_cls_name='TextDataset')
        data_module.setup(stage='fit')
        assert data_module.train_set is not None
        assert data_module.val_set is not None

    def test_loads_test_dataset_correctly(self, mock_load_data):
        data_module = TextDataModule(batch_size=32, num_workers=4, dataset_cls_name='TextDataset')
        data_module.setup(stage='test')
        assert data_module.test_set is not None

    def test_raises_error_for_invalid_dataset_cls_name(self):
        with pytest.raises(AttributeError):
            TextDataModule(batch_size=32, num_workers=4, dataset_cls_name='InvalidDataset')
