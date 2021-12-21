#Comp550 Final Project source code
@Zeying Tian @Yining Duan @Dailun Li


This repository is created based on @Keon's  barebone seq2seq implementation.
## Recurrent neural network classifier with self-attention

This is a a minimal RNN-based classification model (many-to-one) with self-attention.
This script requires the torchtext version 0.11.1. Deprecated versions are not guaranteed to be supported.

#### Model description:
- LSTM and GRU encoder for the embedded input sequence can be found in model.py
- [Scaled dot-product](https://arxiv.org/pdf/1706.03762.pdf) self-attention with the encoder outputs as keys and values and the hidden state as the query
- Logistic regression classifier on top of attention outputs

#### Arguments:

```
  --data DATA        Corpus: [SST, TREC, IMDB, YELP]
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
  --cuda             bool: default True = use CUDA else CPU
  --fine             use fine grained labels in SST # currently unused
```

#### Running experiments:

- initialize a new environment and download the requirements use
  
```
pip install -r requirements.txt
```
- launch the terminal in the current directory, with the following command:
```
python main.py --<data> --<emsize> --<hidden>  --<nlayers>  --<lr> --<clip> --<epochs> --<drop> --<batch_size> --<model> --<bi>
```
For the experiments on no attention, simply modify the model.py script. Adjust 
the classifier class to link the encoder directly with the decoder. 
An example can be found in run.sh
