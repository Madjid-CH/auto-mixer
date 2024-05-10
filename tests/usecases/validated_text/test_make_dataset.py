from pathlib import Path

import pytest

from m2_mixer.usecases.text_db.make_dataset import (mask_label_from_text,
                                                    full_word_from_tokens,
                                                    find_words_derived_from_stem,
                                                    tokenizer,
                                                    MASK, MAX_SEQ_LEN, get_vocabs, get_word_to_index,
                                                    get_index_to_word, get_w2v_embeddings)
from m2_mixer.utils.utils import get_env_var


def test_mask_label_from_text():
    row = {'label': 'run', 'text': 'I like running, runners and runs . and blah blah blah'}
    masked_text = mask_label_from_text(row)
    assert masked_text == f'i like {MASK}, {MASK} and {MASK}. and blah blah blah'


def test_full_word_from_tokens():
    tokens = ['[CLS]', 'mad', '##ori', '##h', '##ka', '##j', '##hi', 'm', '##ji', '##jm', '[SEP]']
    word_beginning_index = 7
    assert full_word_from_tokens(tokens, word_beginning_index) == 'mjijm'


def test_find_words_derived_from_stem():
    tokens = tokenizer("madorihkajhi mjijm " * 3, return_tensors='pt')['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    results = find_words_derived_from_stem(tokens, 'm')
    assert results == ['mjijm', 'mjijm', 'mjijm']


@pytest.fixture
def data():
    return [
        {'label': 'run', 'text': 'I like running, runners and runs . and blah blah blah'},
        {'label': 'dog', 'text': 'I like dogs, doggy and dog . and blah blah blah'},
    ]


@pytest.fixture
def vocabs(data):
    return get_vocabs(data)


def test_get_vocabs(data, vocabs):
    assert len(vocabs) == 14


def test_get_word_to_index(vocabs):
    assert set(get_word_to_index(vocabs).keys()) == set(vocabs)


def test_get_index_to_word(vocabs):
    assert set(get_index_to_word(vocabs).values()) == set(vocabs)


def word2vec_weights_exists():
    if get_env_var('WORD2VEC_PATH') is None:
        return False
    return Path(get_env_var('WORD2VEC_PATH')).exists()


@pytest.mark.skipif(not word2vec_weights_exists(), reason="Word2Vec weights not found")
def test_get_word2vec_embeddings(data):
    embeddings = get_w2v_embeddings(data[0])
    assert embeddings.shape == (MAX_SEQ_LEN, 300)
