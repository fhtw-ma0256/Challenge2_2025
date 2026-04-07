import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import datetime
from tqdm import tqdm
import sys

from models.model_classifier import AudioMLP
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50, get_global_stats
import config

# seed for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# evaluate model on different testing data 'dataloader'
def test(model, dataloader, criterion, device):
    model.eval()

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():
        # no gradient computation needed
        for k, x, label in tqdm(dataloader, unit='bat', disable=config.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            # the forward pass through the model
            y_prob = model(x)

            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch(model, train_loader, criterion, optimizer, device):
    # switch to training
    model.train()

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=config.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        loss = criterion(y_prob, y_true)
        # reset the gradients to zero - avoids accumulation
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        losses.append(loss.item())
        # minimize the loss via the gradient - adapts the model parameters
        optimizer.step()

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, experiment, device):
    num_epochs = config.epochs

    loss_stopping = EarlyStopping(patience=config.patience, delta=0.002, verbose=True, float_fmt=config.float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        # iterate once over training data
        train_acc, train_loss = train_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device)

        # validate model
        val_acc, val_loss, _ = test(model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        # print('\n')
        pbar.update()
        # pbar.refresh() syncs output when pbar on stderr
        # pbar.refresh()
        print(end=' ')
        print(  # f" Epoch: {epoch}/{num_epochs}",
            f"TrnAcc={train_acc:{config.float_fmt}}",
            f"ValAcc={val_acc:{config.float_fmt}}",
            f"TrnLoss={np.mean(train_loss):{config.float_fmt}}",
            f"ValLoss={val_loss_avg:{config.float_fmt}}",
            end=' ')

        early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()
        if early_stop:
            print("Early stopping")
            break

        # advance the optimization scheduler
        scheduler.step()
    # save full model
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


# build model from configuration.
def make_model():
    n = config.n_classes
    model_constructor = config.model_constructor
    print(model_constructor)
    model = eval(model_constructor)
    return model


def train_single_fold(train_set, val_set, test_set, test_fold, experiment_root, device=None, args=None):
    # model, optimizer, scheduler, fit, test
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")

    # save results to args.output_dir/fold_id/
    experiment = os.path.join(experiment_root, f'{test_fold}')
    if not os.path.exists(experiment):
        os.mkdir(experiment)

    # clone stdout to file (does not include stderr). If used may confuse linux 'tee' command.
    with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
             newline='\n', proc_cr=True):
        print('*****')
        print(f'experiment: {experiment}')
        print(f'train folds: {train_set.train_folds}')
        print(f'test fold: {train_set.test_folds}')
        print('random wave cropping')

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=False,
            persistent_workers=config.persistent_workers,
            pin_memory=True,
            prefetch_factor=config.prefetch_factor,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )

        print()
        # instantiate model
        model = make_model()
        # model = nn.DataParallel(model, device_ids=config.device_ids)
        model = model.to(device)
        print('*****')

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=0.9,
                                    weight_decay=config.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.step_size,
                                                    gamma=config.gamma)

        # fit the model using only training and validation data, no testing data allowed here
        print()
        fit_classifier(model, train_loader, val_loader,
                       criterion=criterion, optimizer=optimizer, scheduler=scheduler, experiment=experiment,
                       device=device)

        # tests
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,  # config.num_workers,
                                                  drop_last=False,
                                                  )

        print(f'\ntest {experiment}')
        test_acc, test_loss, _ = test(model, test_loader, criterion=criterion, device=device)
        scores = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Test fold (1-5)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results/single_fold')
    args = parser.parse_args()

    # Override config if provided
    for key, value in vars(args).items():
        if value is not None:
            #assert hasattr(config, key), f"Unknown config parameter: {key}"
            setattr(config, key, value)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load stats and dataset for the specified fold
    global_stats = get_global_stats(config.esc50_path)

    train_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                      subset="train", global_mean_std=global_stats[args.fold - 1])
    val_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                    subset="val", global_mean_std=global_stats[args.fold - 1])
    test_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                     subset="test", global_mean_std=global_stats[args.fold - 1])

    result = train_single_fold(train_set, val_set, test_set, args.fold,
                               args.output_dir, args=args)
    print(result)
