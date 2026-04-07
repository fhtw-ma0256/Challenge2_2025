import torch
import pandas as pd
import datetime
import os
from functools import partial

from dataset.dataset_ESC50 import ESC50, get_global_stats
from train import train_single_fold
import config


if __name__ == "__main__":
    data_path = config.esc50_path

    pd.options.display.float_format = ('{:,' + config.float_fmt + '}').format
    runs_path = config.runs_path
    experiment_root = os.path.join(runs_path, str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    os.makedirs(experiment_root, exist_ok=True)

    # Load normalization stats (auto-switches between hardcoded and computed)
    global_stats = get_global_stats(data_path)
    # for all folds
    scores = {}
    for test_fold in config.test_folds:
        # this function assures consistent 'test_folds' setting for train, val, test splits
        get_fold_dataset = partial(ESC50, root=data_path, download=True,
                                   test_folds={test_fold}, global_mean_std=global_stats[test_fold - 1])

        scores[test_fold] = train_single_fold(train_set=get_fold_dataset(subset="train"),
                                              val_set=get_fold_dataset(subset="val"),
                                              test_set=get_fold_dataset(subset="test"), test_fold=test_fold,
                                              experiment_root=experiment_root)
        print(scores[test_fold])
        # print(scores[test_fold].unstack())
        print()

    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(['mean', 'std']))))
