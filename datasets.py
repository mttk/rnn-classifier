from torchtext.legacy import data
from torchtext.legacy import datasets


def make_sst(batch_size, device=-1, fine_grained=False, vectors=None):
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
        (train, val, test), batch_size=batch_size, device=device, repeat=False)

    return (train_iter, val_iter, test_iter), text_field, label_field


def make_imdb(batch_size, device=-1, vectors=None):
    text_field = data.Field(include_lengths=True, lower=True)
    label_field = data.LabelField()
    train, test = datasets.IMDB.splits(text_field, label_field)

    text_field.build_vocab(train, test, vectors=vectors, max_size=30000)
    label_field.build_vocab(train, test)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, device=device, repeat=False)

    return (train_iter, test_iter), text_field, label_field


def make_trec(batch_size, device=-1, vectors=None):
    text_field = data.Field(include_lengths=True, lower=True)
    label_field = data.LabelField()
    train, test = datasets.TREC.splits(text_field, label_field)

    text_field.build_vocab(train, test, vectors=vectors)
    label_field.build_vocab(train, test)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, device=device, repeat=False)

    return (train_iter, test_iter), text_field, label_field


dataset_map = {
    'SST': make_sst,
    'IMDB': make_imdb,
    'TREC': make_trec
}
