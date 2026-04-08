"""
Single-fold training script for ESC-50 audio classification.

Can be used in two ways:
1. Imported by train_crossval.py for cross-validation
2. Run directly for hyperparameter experimentation:
   python train.py --fold 1 --lr 0.01 --epochs 100
"""

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

# ========== REPRODUCIBILITY ==========
# Fixed seeds ensure identical results across runs
# Critical for debugging and comparing experiments
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)  # Enforces deterministic GPU ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Disable auto-tuning (deterministic > speed)

def test(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset (validation or test).

    Returns:
        acc: Classification accuracy (0.0 to 1.0)
        losses: List of batch losses
        probs: Dict mapping filenames to class probabilities
    """
    model.eval()  # Disable dropout, batch norm uses running stats

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():  # No gradient computation = faster, less memory
        for k, x, label in tqdm(dataloader, unit='bat', disable=config.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            y_prob = model(x)  # Forward pass

            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)  # Highest probability class
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]

            # Store probabilities for each file (useful for analysis/voting)
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch over the entire training set.

    Returns:
        acc: Training accuracy for this epoch
        losses: List of batch losses
    """
    model.train()  # Enable dropout, batch norm uses batch stats

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=config.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        y_prob = model(x)
        loss = criterion(y_prob, y_true)

        # Standard training steps:
        optimizer.zero_grad()  # 1. Clear old gradients
        loss.backward()  # 2. Compute new gradients via backprop
        optimizer.step()  # 3. Update model parameters

        losses.append(loss.item())

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, experiment, device):
    """
    Main training loop with early stopping and checkpointing.
    """
    num_epochs = config.epochs

    # Early stopping monitors validation loss, saves best model
    loss_stopping = EarlyStopping(patience=config.patience, delta=0.002, verbose=True,
                                  float_fmt=config.float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in range(1, 1 + num_epochs):
        # Train for one epoch
        train_acc, train_loss = train_epoch(model, train_loader, criterion=criterion,
                                            optimizer=optimizer, device=device)

        # Validate after each epoch
        val_acc, val_loss, _ = test(model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        pbar.update()
        print(end=' ')
        print(f"TrnAcc={train_acc:{config.float_fmt}}",
              f"ValAcc={val_acc:{config.float_fmt}}",
              f"TrnLoss={np.mean(train_loss):{config.float_fmt}}",
              f"ValLoss={val_loss_avg:{config.float_fmt}}",
              end=' ')

        # Early stopping check
        early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()  # Newline when no improvement (no checkpoint saved)
        if early_stop:
            print("Early stopping")
            break

        scheduler.step()  # Decay learning rate (StepLR: every step_size epochs)

    # Save final model (not necessarily the best one)
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


def make_model():
    """Instantiate model from config.model_constructor string."""
    model_constructor = config.model_constructor
    print(f"Building model: {model_constructor}")
    model = eval(model_constructor)  # Safe because config is controlled
    return model


def train_single_fold(train_set, val_set, test_set, test_fold, experiment_root, device=None, args=None):
    """
    Complete training pipeline for a single fold.

    Args:
        train_set, val_set, test_set: ESC50 datasets with proper splits
        test_fold: Fold number (1-5) used as test set
        experiment_root: Parent directory for saving results
        device: Torch device (auto-detected if None)
        args: Optional argparse namespace with hyperparameter overrides

    Returns:
        pd.Series with TestAcc and TestLoss
    """
    # Auto-detect device if not provided
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")

    # Create fold-specific experiment directory
    experiment = os.path.join(experiment_root, f'{test_fold}')
    os.makedirs(experiment, exist_ok=True)

    # Tee duplicates stdout to a log file (for later review)
    with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
             newline='\n', proc_cr=True):
        print('*****')
        print(f'experiment: {experiment}')
        print(f'train folds: {train_set.train_folds}')
        print(f'test fold: {train_set.test_folds}')
        print('random wave cropping')

        # DataLoaders with prefetching for efficiency
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,  # Important for training to avoid order bias
            num_workers=config.num_workers,
            drop_last=False,
            persistent_workers=config.persistent_workers,
            pin_memory=True,  # Speeds up CPU->GPU transfer
            prefetch_factor=config.prefetch_factor,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,  # No shuffling for validation (deterministic)
            num_workers=config.num_workers,
            drop_last=False,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )

        print()
        model = make_model()
        model = model.to(device)
        print('*****')

        criterion = nn.CrossEntropyLoss().to(device)

        # SGD with momentum is standard for image/audio classification
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=0.9,
                                    weight_decay=config.weight_decay)  # L2 regularization

        # StepLR: lr = lr * gamma every step_size epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.step_size,
                                                    gamma=config.gamma)

        print()
        fit_classifier(model, train_loader, val_loader,
                       criterion=criterion, optimizer=optimizer,
                       scheduler=scheduler, experiment=experiment, device=device)

        # Final evaluation on held-out test set
        test_loader = DataLoader(test_set,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False)

        print(f'\ntest {experiment}')
        test_acc, test_loss, _ = test(model, test_loader, criterion=criterion, device=device)
        scores = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
        return scores


if __name__ == "__main__":
    """
    Standalone mode: Train a single fold with custom hyperparameters.
    Example: python train.py --fold 1 --lr 0.01 --epochs 100
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Test fold (1-5)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results/single_fold')
    args = parser.parse_args()

    # Apply command-line overrides to config module
    # Note: Only overrides if value is provided (not None)
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load normalization statistics (auto-selects hardcoded or computes)
    global_stats = get_global_stats(config.esc50_path)

    # Create train/val/test splits for the specified fold
    train_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                      subset="train", global_mean_std=global_stats[args.fold - 1])
    val_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                    subset="val", global_mean_std=global_stats[args.fold - 1])
    test_set = ESC50(root=config.esc50_path, test_folds={args.fold},
                     subset="test", global_mean_std=global_stats[args.fold - 1])

    # Train and evaluate
    result = train_single_fold(train_set, val_set, test_set, args.fold,
                               args.output_dir, args=args)
    print(result)