import h5py
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

DATASET_PATH = '../data/mm_imdb/multimodal_imdb.hdf5'
METADATA_PATH = "../data/mm_imdb/metadata.npy"
METADATA = np.load(METADATA_PATH, allow_pickle=True).item()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'bert-base-uncased'

BERT_MAX_INPUT_LENGTH = 512
EMBEDDING_DIM = 768
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, padding_side='right', padding_max_length=BERT_MAX_INPUT_LENGTH,
                                          truncation=True, max_length=BERT_MAX_INPUT_LENGTH)
model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)


def make_text_embeddings():
    with h5py.File(DATASET_PATH, 'r+') as full_dataset:
        sequences = full_dataset["sequences"]
        if "embeddings" in full_dataset:
            del full_dataset["embeddings"]
        embeddings_dataset = full_dataset.create_dataset("embeddings",
                                                         (sequences.shape[0],
                                                          BERT_MAX_INPUT_LENGTH,
                                                          EMBEDDING_DIM),
                                                         dtype=float)
        for i, sequence in enumerate(sequences):
            sample_text = generate_text_from_sequence(sequence, METADATA['ix_to_word'])
            encoded_input = tokenizer(sample_text, return_tensors='pt', truncation=True, padding="max_length",
                                      max_length=BERT_MAX_INPUT_LENGTH)
            encoded_input = to_device(encoded_input)
            with torch.no_grad():
                model_output = model(**encoded_input)
            embeddings = model_output.last_hidden_state
            embeddings_dataset[i] = embeddings.cpu()

            if i % 100 == 0:
                print(f"Processed {i} samples")


def to_device(encoded_input):
    return {key: val.to(DEVICE) for key, val in encoded_input.items()}


def generate_text_from_sequence(sequence, lookup):
    return ' '.join([lookup[word] for word in sequence])


if __name__ == '__main__':
    make_text_embeddings()
