"""
@date: 2021-12-17

@brief
This file consists of utils to download datasets.

Currently there are three available datasets:
1. SST
2. IMDB
3. TREC
4.

All of the above three datasets are available from
`torchtext.datasets`.

For each of the util function, we shall return three
things.
1. A tuple of dataset iterators.
2. The text `Field` of the data.
3. The label `Field` of the data.

"""

import pandas as pd
from pathlib import Path
from torchtext.legacy import data
from torchtext.legacy.data.iterator import Iterator
from torchtext.legacy import datasets
from torch import device
from torchtext.legacy.data import Field
from typing import Callable, Tuple, Dict, Any, List

dataset_func = Callable[[int, device, str, Any], Tuple[Tuple, Field, Field]]


def make_sst(
        batch_size: int,
        operating_device: device,
        vectors=None,
        fine_grained=False,
):
    text_field = data.Field(include_lengths=True, lower=True)
    label_field = data.LabelField()
    f = lambda ex: ex.label != 'neutral' if not fine_grained else lambda ex: True
    train, val, test = datasets.SST.splits(
        text_field, label_field,
        fine_grained=fine_grained,
        train_subtrees=False,
        filter_pred=f
    )

    text_field.build_vocab(train, test, val, vectors=vectors)
    label_field.build_vocab(train, test, val)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=operating_device, repeat=False)

    return (train_iter, val_iter, test_iter), text_field, label_field


def make_imdb(
        batch_size: int,
        operating_device: device,
        vectors=None
):
    text_field = data.Field(include_lengths=True, lower=True)
    label_field = data.LabelField()
    train, test = datasets.IMDB.splits(text_field, label_field)

    text_field.build_vocab(train, test, vectors=vectors, max_size=30000)
    label_field.build_vocab(train, test)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, device=operating_device, repeat=False)

    return (train_iter, test_iter), text_field, label_field


def make_trec(
        batch_size,
        operating_device=-1,
        vectors=None
):
    text_field = data.Field(include_lengths=True, lower=True)
    label_field = data.LabelField()
    train, test = datasets.TREC.splits(text_field, label_field)

    text_field.build_vocab(train, test, vectors=vectors)
    label_field.build_vocab(train, test)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, device=operating_device, repeat=False)

    return (train_iter, test_iter), text_field, label_field


def make_yelp(
        batch_size,
        operating_device=-1,
        vectors=None,
):
    dataset_path = Path('./yelp_review_polarity_csv')
    # Append labels to the dataset.
    train_dataset_raw = pd.read_csv(dataset_path / 'train.csv')
    test_dataset_raw = pd.read_csv(dataset_path / 'test.csv')
    train_dataset_raw = train_dataset_raw.sample(frac = 1)
    train_dataset_raw = train_dataset_raw[:150000]

    train_dataset_raw.columns = ['label', 'text']
    test_dataset_raw.columns = ['label', 'text']
    train_dataset_raw.to_csv(dataset_path / 'train_labeled.csv', index=False)
    test_dataset_raw.to_csv(dataset_path / 'test_labeled.csv', index=False)

    # Start processing.
    text_field = Field(
        include_lengths=True,
        lower=True
    )
    label_field = data.LabelField()

    train_obj, test_obj = data.TabularDataset.splits(
        path=str(dataset_path),
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=[('label', label_field), ('text', text_field)]
    )

    text_field.build_vocab(train_obj, test_obj, vectors=vectors)
    label_field.build_vocab(train_obj, test_obj)

    train_iter, test_iter = data.BucketIterator.splits(
        (train_obj, test_obj),
        batch_size=batch_size,
        device=operating_device,
        repeat=False,
        sort=False
    )

    return (train_iter, test_iter), text_field, label_field


def mean_sentence_length(text_iter: Iterator) -> float:
    len_list: List[int] = list(map(lambda item: len(item.text), text_iter.data()))
    return sum(len_list) / len(len_list)


dataset_map: Dict[str, dataset_func] = {
    'SST': make_sst,
    'IMDB': make_imdb,
    'TREC': make_trec,
    "YELP": make_yelp
}
