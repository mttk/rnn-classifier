import torch
from simple_chalk import chalk
from torch.nn import Embedding
from typing import List, Any, Dict
from pathlib import Path
from model import *
from train import *
import os
import json
from datasets import dataset_map

pretrained_GloVe_sizes = [50, 100, 200, 300]


def seed_everything(seed, use_gpu=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)


def load_pretrained_vectors(dim):
    if dim in pretrained_GloVe_sizes:
        # Check torchtext.datasets.vocab line #383
        # for other pretrained vectors. 6B used here
        # for simplicity
        name = 'glove.{}.{}d'.format('6B', str(dim))
        return name
    return None


def save_state(
        report: List[Dict[str, Any]],
        epoch_num: int,
        m: Module,
        result_folder: Path
) -> None:
    cur_epoch_folder = result_folder / "epoch-{}".format(epoch_num)

    # Make the result folder.
    os.mkdir(cur_epoch_folder)

    # Save the report and model to the current epoch folder.
    with open(cur_epoch_folder / "report.json", 'w') as r:
        json.dump(report, r, indent=4)

    # Save the model
    torch.save(m.state_dict(), cur_epoch_folder / "model.pt")


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
    parser.add_argument('--output_path', type=str, default='report',
                        help='Output directory.')
    return parser


if __name__ == "__main__":
    # Accept CLI arguments.
    args = make_parser().parse_args()
    print(f'{chalk.bold(chalk.greenBright("[Model Hyper-parameters]:"))} {"{}".format(str(args))}')

    # Decide the platform to run.
    cuda = torch.cuda.is_available() and args.cuda
    report_path = args.output_path
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, use_gpu=cuda)
    vectors = load_pretrained_vectors(args.emsize)

    # Load dataset iterators
    data_iter, text_field, label_field = dataset_map[args.data](
        args.batch_size,
        operating_device=device,
        vectors=vectors
    )

    # Some datasets just have the train & test sets, so we just pretend test is valid
    if len(data_iter) == 3:
        train_iter, val_iter, test_iter = data_iter
    else:
        train_iter, val_iter = data_iter
        test_iter = val_iter

    print(f'{chalk.bold(chalk.yellowBright("[Corpus]:"))} '
          f'{"train: {}, test: {}, vocab: {}, labels: {}".format(len(train_iter.dataset), len(test_iter.dataset), len(text_field.vocab), len(label_field.vocab))}')

    num_tokens, num_labels = len(text_field.vocab), len(label_field.vocab)
    args.nlabels = num_labels  # hack to not clutter function arguments

    embedding = Embedding(num_tokens, args.emsize, padding_idx=1, max_norm=1)
    if vectors:
        embedding.weight.data.copy_(text_field.vocab.vectors)

    encoder = Encoder(
        args.emsize,
        args.hidden,
        nlayers=args.nlayers,
        dropout=args.drop,
        bidirectional=args.bi,
        rnn_type=args.model
    )

    attention_dim = args.hidden if not args.bi else 2 * args.hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = Classifier(embedding, encoder, attention, attention_dim, num_labels)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    training_optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

    # Start training.
    training_report: List[Dict[str, Any]] = list()
    for epoch in range(1, args.epochs + 1):
        training_result = train_an_epoch(model, train_iter, training_optimizer, criterion, args, epoch)
        val_result = evaluate_an_epoch(model, val_iter, criterion, args, epoch)
        training_report.append({
            "train": training_result,
            "validation": val_result
        })
        save_state(training_report, epoch, model, report_path)

    val_loss = evaluate_an_epoch(model, test_iter, criterion, args, 0)
