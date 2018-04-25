import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn

from datasets import dataset_map
from model import *
from torchtext.vocab import GloVe

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='SST',
                        help='Data corpus: [SST, TREC, IMDB]')
  parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
  parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
  parser.add_argument('--hidden', type=int, default=500,
                        help='number of hidden units for the RNN encoder')
  parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers of the RNN encoder')
  parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
  parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
  parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
  parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
  parser.add_argument('--cuda', action='store_false',
                    help='[DONT] use CUDA')
  parser.add_argument('--fine', action='store_true', 
                    help='use fine grained labels in SST')
  return parser


def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)


def update_stats(accuracy, confusion_matrix, logits, y):
  _, max_ind = torch.max(logits, 1)
  equal = torch.eq(max_ind, y)
  correct = int(torch.sum(equal))

  for j, i in zip(max_ind, y):
    confusion_matrix[int(i),int(j)]+=1

  return accuracy + correct, confusion_matrix


def train(model, data, optimizer, criterion, args):
  model.train()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    x, lens = batch.text
    y = batch.label

    logits, _ = model(x)
    loss = criterion(logits.view(-1, args.nlabels), y)
    total_loss += float(loss)
    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r', flush=True)
    t = time.time()

  print()
  print("[Loss]: {:.5f}".format(total_loss / len(data)))
  print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  print(confusion_matrix)
  return total_loss / len(data)


def evaluate(model, data, optimizer, criterion, args, type='Valid'):
  model.eval()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      x, lens = batch.text
      y = batch.label

      logits, _ = model(x)
      total_loss += float(criterion(logits.view(-1, args.nlabels), y))
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)
      t = time.time()

  print()
  print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
  print("[{} accuracy]: {}/{} : {:.3f}%".format(type,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  print(confusion_matrix)
  return total_loss / len(data)

pretrained_GloVe_sizes = [50, 100, 200, 300]

def load_pretrained_vectors(dim):
  if dim in pretrained_GloVe_sizes:
    # Check torchtext.datasets.vocab line #383
    # for other pretrained vectors. 6B used here
    # for simplicity
    name = 'glove.{}.{}d'.format('6B', str(dim))
    return name
  return None

def main():
  args = make_parser().parse_args()
  print("[Model hyperparams]: {}".format(str(args)))

  cuda = torch.cuda.is_available() and args.cuda
  device = torch.device("cpu") if not cuda else torch.device("cuda:0")
  seed_everything(seed=1337, cuda=cuda)
  vectors = load_pretrained_vectors(args.emsize)

  # Load dataset iterators
  iters, TEXT, LABEL = dataset_map[args.data](args.batch_size, device=device, vectors=vectors)

  # Some datasets just have the train & test sets, so we just pretend test is valid
  if len(iters) == 3:
    train_iter, val_iter, test_iter = iters
  else:
    train_iter, test_iter = iters
    val_iter = test_iter

  print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
            len(train_iter.dataset), len(test_iter.dataset), len(TEXT.vocab), len(LABEL.vocab)))

  ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)
  args.nlabels = nlabels # hack to not clutter function arguments

  embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1, max_norm=1)
  if vectors: embedding.weight.data.copy_(TEXT.vocab.vectors)
  encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers, 
                    dropout=args.drop, bidirectional=args.bi, rnn_type=args.model)

  attention_dim = args.hidden if not args.bi else 2*args.hidden
  attention = Attention(attention_dim, attention_dim, attention_dim)

  model = Classifier(embedding, encoder, attention, attention_dim, nlabels)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

  try:
    best_valid_loss = None

    for epoch in range(1, args.epochs + 1):
      train(model, train_iter, optimizer, criterion, args)
      loss = evaluate(model, val_iter, optimizer, criterion, args)

      if not best_valid_loss or loss < best_valid_loss:
        best_valid_loss = loss

  except KeyboardInterrupt:
    print("[Ctrl+C] Training stopped!")
  loss = evaluate(model, test_iter, optimizer, criterion, args, type='Test')

if __name__ == '__main__':
  main()