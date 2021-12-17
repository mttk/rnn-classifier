import argparse
import time
import numpy as np
from torch.nn import Module
from torch import max, eq, sum, no_grad
from torch.nn.utils import clip_grad_norm_
from torch.nn.modules import loss
from torch.optim import optimizer
from torchtext.legacy.data.iterator import Iterator


def update_stats(correct_predictions, confusion_matrix, logit, y):
    _, max_ind = max(logit, 1)
    equal = eq(max_ind, y)
    correct = int(sum(equal))

    for j, i in zip(max_ind, y):
        confusion_matrix[int(i), int(j)] += 1

    return correct_predictions + correct, confusion_matrix


def train_an_epoch(
        model: Module,
        data: Iterator,
        opt: optimizer,
        criterion: loss,
        args: argparse
):
    model.train()
    correct_predictions, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
    t = time.time()
    total_loss = 0
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, lens = batch.text
        y = batch.label

        logit, _ = model(x)
        curr_loss = criterion(logit.view(-1, args.nlabels), y)
        total_loss += curr_loss
        correct_predictions, confusion_matrix = update_stats(correct_predictions, confusion_matrix, logit, y)
        curr_loss.backward()
        clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)
        t = time.time()

    print()
    print("[Loss]: {:.5f}".format(total_loss / len(data)))
    print("[Accuracy]: {}/{} : {:.3f}%".format(correct_predictions, len(data.dataset),
                                               correct_predictions / len(data.dataset) * 100))
    print(confusion_matrix)
    return total_loss / len(data)


def evaluate_an_epoch(
        model: Module,
        data: Iterator,
        criterion: loss,
        args: argparse,
        metric_label='Validation'
):
    model.eval()
    correct_predictions, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
    t = time.time()
    total_loss = 0
    with no_grad():
        for batch_num, batch in enumerate(data):
            x, lens = batch.text
            y = batch.label

            logit, _ = model(x)
            total_loss += float(criterion(logit.view(-1, args.nlabels), y))
            correct_predictions, confusion_matrix = update_stats(correct_predictions, confusion_matrix, logit, y)
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    print("[{} loss]: {:.5f}".format(metric_label, total_loss / len(data)))
    print("[{} accuracy]: {}/{} : {:.3f}%".format(metric_label,
                                                  correct_predictions, len(data.dataset),
                                                  correct_predictions / len(data.dataset) * 100))
    print(confusion_matrix)
    return total_loss / len(data)
