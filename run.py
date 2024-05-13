import wandb

from auto_mixer.datasets.mm_imdb.mmimdb import MMIMDBDataModule
from auto_mixer.runner import train


def main():
    wandb.init(project='auto-mixer', name='test', mode='disabled')
    dataloader = MMIMDBDataModule(batch_size=32, num_workers=4, max_seq_len=512,
                                  dataset_cls_name="MMIMDBDatasetWithEmbeddings")
    dataloader.setup()
    train(dataloader)


if __name__ == '__main__':
    main()
