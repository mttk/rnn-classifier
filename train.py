import argparse
import time
import numpy as np
from simple_chalk import chalk
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
        args: argparse,
        epoch_num: int
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

    epoch_loss, epoch_acc = total_loss / len(data), correct_predictions / len(data.dataset) * 100
    time_elapsed = time.time() - t

    print(f'{chalk.bold("TRAINING   --->")} '
          f'{chalk.bold.greenBright("[EPOCH-{}]".format(epoch_num))} '
          f'{chalk.bold("TIME ELAPSED:")} {"{:.2f}s".format(time_elapsed)} '
          f'{chalk.bold.redBright("LOSS:")} {"{:.5f}".format(epoch_loss)} '
          f'{chalk.bold.yellowBright("ACC:")} {"{:.2f}".format(epoch_acc)}')

    return {
        "epoch_num": epoch_num,
        "epoch_loss:": epoch_loss.item(),
        "epoch_acc": epoch_acc,
        "time_elapsed": time_elapsed,
        "confusion_matrix": confusion_matrix.tolist()
    }


def evaluate_an_epoch(
        model: Module,
        data: Iterator,
        criterion: loss,
        args: argparse,
        epoch_num: int = 1
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

    epoch_loss, epoch_acc = total_loss / len(data), correct_predictions / len(data.dataset) * 100
    time_elapsed = time.time() - t

    print(f'{chalk.bold("VALIDATION --->")} '
          f'{chalk.bold.greenBright("[EPOCH-{}]".format(epoch_num))} '
          f'{chalk.bold("TIME ELAPSED:")} {"{:.2f}s".format(time_elapsed)} '
          f'{chalk.bold.redBright("LOSS:")} {"{:.5f}".format(epoch_loss)} '
          f'{chalk.bold.yellowBright("ACC:")} {"{:.2f}".format(epoch_acc)}')

    return {
        "epoch_num": epoch_num,
        "epoch_loss": epoch_loss.item(),
        "epoch_acc": epoch_acc,
        "time_elapsed": time_elapsed,
        "confusion_matrix": confusion_matrix.tolist()
    }
