## Recurrent neural network classifier with self-attention

A minimal RNN-based classification model (many-to-one) with self-attention.
Tested on `master` branches of both `torch` (commit 5edf6b2) and `torchtext` (commit c839a79). The `volatile` warnings that might be printed are due to using pytorch version 4 with torchtext.

Inspired by @Keon's [barebone seq2seq implementation](https://github.com/keon/seq2seq), this repository aims to provide a minimal implementation of an RNN classifier with self-attention.

#### Model description:
- LSTM or GRU encoder for the embedded input sequence
- [Scaled dot-product](https://arxiv.org/pdf/1706.03762.pdf) self-attention with the encoder outputs as keys and values and the hidden state as the query
- Logistic regression classifier on top of attention outputs

#### Arguments:

```
  --data DATA        Corpus: [SST, TREC, IMDB]
  --model MODEL      type of recurrent net [LSTM, GRU]
  --emsize EMSIZE    size of word embeddings [Uses pretrained on 50, 100, 200, 300]
  --hidden HIDDEN    number of hidden units for the RNN encoder
  --nlayers NLAYERS  number of layers of the RNN encoder
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --drop DROP        dropout
  --bi               bidirectional encoder
  --cuda             [DONT] use CUDA
  --fine             use fine grained labels in SST # currently unused
```

A sample set of arguments can be viewed in `run.sh`.

#### Results 

Accuracy on test set after 5 epochs of the model with sample params:

|               |    SST    |    TREC   |    IMDB   |
| ------------- |:---------:|:---------:|:---------:|
| `run.sh`      |  80.340%  |  87.000%  |  86.240%  |
