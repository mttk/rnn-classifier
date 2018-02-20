from torchtext import data
from torchtext import datasets


def make_sst(batch_size, device=-1, fine_grained=False, vectors=None):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  filter_pred = lambda ex: ex.label != 'neutral' if not fine_grained else lambda ex: True
  train, val, test = datasets.SST.splits(TEXT, LABEL, 
                                         fine_grained=fine_grained, 
                                         train_subtrees=False,
                                         filter_pred=filter_pred
                                         )

  TEXT.build_vocab(train, vectors=vectors)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter = data.BucketIterator.splits(
              (train, val, test), batch_size=batch_size, device=device, repeat=False)

  return (train_iter, val_iter, test_iter), TEXT, LABEL


def make_imdb(batch_size, device=-1, vectors=None):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  train, test = datasets.IMDB.splits(TEXT, LABEL)

  TEXT.build_vocab(train, vectors=vectors, max_size=30000) 
  LABEL.build_vocab(train)
  train_iter, test_iter = data.BucketIterator.splits(
              (train, test), batch_size=batch_size, device=device, repeat=False)

  return (train_iter, test_iter), TEXT, LABEL


def make_trec(batch_size, device=-1, vectors=None):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  train, test = datasets.TREC.splits(TEXT, LABEL)

  TEXT.build_vocab(train, vectors=vectors)
  LABEL.build_vocab(train)
  train_iter, test_iter = data.BucketIterator.splits(
              (train, test), batch_size=batch_size, device=device, repeat=False)

  return (train_iter, test_iter), TEXT, LABEL


dataset_map = {
  'SST' : make_sst,
  'IMDB' : make_imdb,
  'TREC' : make_trec
}


if __name__ == '__main__':
  (tr, val, te), T, L = make_sst(20)
  print("[SST] vocab: {} labels: {}".format(len(T.vocab), len(L.vocab)))
  print("[SST] train: {} val: {} test {}".format(len(tr.dataset), len(val.dataset), len(te.dataset)))

  (tr, te), T, L = make_imdb(20)
  print("[IMDB] vocab: {} labels: {}".format(len(T.vocab), len(L.vocab)))
  print("[IMDB] train: {} test {}".format(len(tr.dataset), len(te.dataset)))

  (tr, te), T, L = make_trec(20)
  print("[TREC] vocab: {} labels: {}".format(len(T.vocab), len(L.vocab)))
  print("[TREC] train: {} test {}".format(len(tr.dataset), len(te.dataset)))
