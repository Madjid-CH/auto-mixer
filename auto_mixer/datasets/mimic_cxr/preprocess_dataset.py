import re

import pandas as pd
import polars as pl
import torch
from transformers import AutoTokenizer
from transformers import BertModel

# ROOT_DIR = "/scratch/achergui/data/mimic-cxr"
ROOT_DIR = "D:"

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")


# MAX_SEQ_LEN = 301


def normalize(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()  # Replaces HTML line break tags with spaces.
    text = re.sub(r'^https?://.*[\r\n]*', ' L ', text, flags=re.MULTILINE)  # Replaces URLs with a placeholder.
    text = re.sub(r'[~*+^`_#\[\]|]', r' ', text).strip()  # Replaces special characters with spaces.
    text = re.sub(r'[0-9]+', r' N ', text).strip()  # Replaces numbers with a placeholder.
    text = re.sub(r'([/\'\-.?!()",:;])', r' \1 ', text).strip()  # Puts spaces around certain punctuation.
    return text


def tokenized_text(text):
    result = tokenizer(text, return_tensors='pt')['input_ids'][0]
    result = tokenizer.convert_ids_to_tokens(result)
    return result


device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert = BertModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-general').to(device)
bert.eval()


def generate_word_embeddings_with_bert(text):
    encoded_input = tokenizer(text, return_tensors='pt',
                              # padding='max_length', max_length=MAX_SEQ_LEN
                              ).to(device)
    with torch.no_grad():
        model_output = bert(**encoded_input)

    embeddings = model_output.last_hidden_state
    return embeddings


def get_embeddings(row):
    embeddings = generate_word_embeddings_with_bert(row)
    return embeddings.squeeze().cpu().numpy()


def extract_bert_embeddings():
    splits_paths = ['/mimic-cxr-jpg_full_val.tsv',
                    '/mimic-cxr-jpg_full_train.tsv', ]
    embeddings_splits_paths = ['/mimic-cxr-jpg_full_val.pkl',
                               '/mimic-cxr-jpg_full_train.pkl']
    for split_path, embeddings_split_path in zip(splits_paths, embeddings_splits_paths):
        print(split_path)
        split = pl.scan_csv(f'{ROOT_DIR}{split_path}', separator='\t')
        split = split.with_columns(
            pl.col("impression").
            map_elements(get_embeddings,
                         return_dtype=pl.Object).alias("embeddings"),
        )
        split.collect().to_pandas().to_pickle(f'{ROOT_DIR}{embeddings_split_path}')


if __name__ == '__main__':
    # extract_bert_embeddings()
    labels = pd.read_csv(f"{ROOT_DIR}/mimic-cxr-2.0.0-chexpert.csv")
    splits = pd.read_csv(f"{ROOT_DIR}/mimic-cxr-2.0.0-split.csv")
    val_df = pd.read_pickle(f"{ROOT_DIR}/mimic-cxr-jpg_full_val.pkl")
    train_df = pd.read_pickle(f"{ROOT_DIR}/mimic-cxr-jpg_full_train.pkl")
    df = pd.concat([train_df, val_df])

    labels = labels.fillna(0).replace(-1, 0)
    labels['labels'] = labels[['Atelectasis', 'Cardiomegaly',
                               'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                               'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                               'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']].values.tolist()
    labels = labels[['study_id', 'labels']]
    df['study_id'] = df['study'].apply(lambda x: x.replace('s', '')).astype(int)
    df = df.merge(labels, on='study_id', how='left')
    df = df.merge(splits, on='study_id', how='left')
    df.drop(columns=['study_id', 'findings', 'dicom_id', 'subject_id']).to_pickle(f'{ROOT_DIR}/mimic_cxr.pkl')
