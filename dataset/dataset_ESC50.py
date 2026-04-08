import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import librosa

import config
from . import transforms
from .splits_ESC50 import get_esc50_splits  # Fixed splits – DO NOT REMOVE


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    #import wget
    import zipfile
    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):

    def __init__(self, root, test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 1.0), download=False):
        """
        Args:
            root: Path to dataset root
            test_folds: Set containing the test fold (e.g., {1})
            subset: "train", "val", or "test"
            global_mean_std: (mean, std) for normalization
            download: Auto-download if missing
        """
        # === FIXED SPLIT LOGIC – DO NOT MODIFY ===
        audio_path = os.path.join(root, 'ESC-50-master/audio')
        self.root = audio_path
        if not os.path.exists(audio_path) and download:
            self._download_dataset(root)

        # Extract single test fold (assumes set with one element)
        assert len(test_folds) == 1
        self.test_folds = set(test_folds)
        self.train_folds = set(range(1, 6)) - test_folds
        self.subset = subset
        self.file_names = get_esc50_splits(root, next(iter(test_folds)), subset, config.val_size)
        # === STUDENTS CAN MODIFY BELOW THIS LINE ===

        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]

        # Feature parameters
        self.n_mfcc = getattr(config, 'n_mfcc', None)

        # Wave transforms (augmentation)
        out_len = int(((config.sr * 5) // config.hop_length) * config.hop_length)
        train_mode = (subset == "train")

        self.wave_transforms = transforms.Compose(
            torch.Tensor,
            transforms.RandomPadding(out_len=out_len, train=train_mode),
            transforms.RandomCrop(out_len=out_len, train=train_mode)
        )

        self.spec_transforms = transforms.Compose(
            torch.Tensor,
            partial(torch.unsqueeze, dim=0),
        )

    def _download_dataset(self, root):
        """Download ESC-50 if not present (students can modify if needed)."""
        os.makedirs(root, exist_ok=True)
        file_path = os.path.join(root, 'master.zip')
        url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
        download_extract_zip(url, file_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)
        wave, rate = librosa.load(path, sr=config.sr)

        # identifying the label of the sample from its name
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])

        if wave.ndim == 1:
            wave = wave[:, np.newaxis]

        # normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0

        # Remove silent sections
        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]

        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)

        if self.n_mfcc:
            mfcc = librosa.feature.mfcc(y=wave_copy.numpy(),
                                        sr=config.sr,
                                        n_mels=config.n_mels,
                                        n_fft=1024,
                                        hop_length=config.hop_length,
                                        n_mfcc=self.n_mfcc)
            feat = mfcc
        else:
            s = librosa.feature.melspectrogram(y=wave_copy.numpy(),
                                               sr=config.sr,
                                               n_mels=config.n_mels,
                                               n_fft=1024,
                                               hop_length=config.hop_length,
                                               #center=False,
                                               )
            log_s = librosa.power_to_db(s, ref=np.max)

            # masking the spectrograms
            log_s = self.spec_transforms(log_s)

            feat = log_s

        # normalize
        if self.global_mean:
            feat = (feat - self.global_mean) / self.global_std

        return file_name, feat, class_id


def calc_global_stats(data_path):
    res = []
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=True)
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std()))
    return np.array(res)


def get_global_stats(data_path):
    """Return global_stats array for current config.
    Uses hardcoded defaults if params match, otherwise computes."""
    if config.n_mels == 128 and config.hop_length == 512:
        stats = np.array([[-54.364834, 20.853344],
                          [-54.279022, 20.847532],
                          [-54.18343, 20.80387],
                          [-54.223698, 20.798292],
                          [-54.200905, 20.949806]])
        print("Using hardcoded global stats (n_mels=128, hop_length=512)")
    else:
        print("Computing global stats for custom params... (slow)")
        stats = calc_global_stats(data_path)
    return stats
