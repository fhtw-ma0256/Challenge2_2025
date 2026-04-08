# Sound Classification on ESC-50 Dataset

This repository contains code and experiments for sound classification on the ESC-50 dataset. The goal is to classify audio recordings into one of 50 environmental sound classes.

## Dataset

The **ESC-50** dataset consists of 50 environmental sound classes, with each class containing 40 recordings, resulting in a total of 2,000 sound recordings. These classes include sounds such as dog barks, gunshots, and various natural sounds.

- **Dataset link:** [ESC-50 Dataset](https://github.com/karoldvl/ESC-50)

### Random Guessing Accuracy

With 50 classes, the expected random guessing accuracy is approximately **2%** (1/50), assuming each class is equally likely to occur.

## Model Overview

The network used for this sound classification task is a **Multilayer Perceptron (MLP)**, which processes **spectrograms** extracted from the audio files. The model is evaluated using **5-fold cross-validation** based on the predefined splits from the ESC-50 dataset.

### Training Process

- The data is split into 5 folds, ensuring that the training and testing data are split evenly across the different classes.
- For each fold, an MLP model is trained on the training data and tested on the hold-out validation data.
- The model architecture is designed to take in spectrograms as input features.

## Installation

It is recommended to create a new environment. E.g., conda instructions:

```bash
conda create -n challenge2 python=3.12
conda activate challenge2
```

Install the environment packages and clone the repository. If you get dependency errors, try installing pytorch first:

```bash
pip install -r requirements.txt
git clone REPO_URL
cd REPO_NAME
```

## Important: Experimental Protocol

To ensure fair comparison and reproducible results, **do not modify** the following files:

- `train_crossval.py` – Orchestrates the 5‑fold cross‑validation loop.
- `test_crossval.py` – Evaluates trained models on held‑out test sets.
- `dataset/splits.py` – Contains the fixed fold splits.

You are free to experiment with:

- `train.py` – Training logic, hyperparameters, model architecture.
- `dataset/dataset_ESC50.py` – Data augmentation, feature extraction (mel spectrograms, MFCCs), normalization.
- `config.py` – Hyperparameters (learning rate, batch size, etc.).

Modifying the protected files will break the cross‑validation protocol and invalidate your results.  
If you change `n_mels` or `hop_length` in `config.py`, the script will automatically recompute normalization statistics 
(this may take a while). Consider if you need normalisation for the given model architecture. 

# Usage

## Full cross-validation
To run the complete 5‑fold cross‑validation experiment:

```bash
python train_crossval.py
```
This will start the training process using the MLP model and 5-fold cross-validation. Results are automatically saved in 
results/ (created if missing). For each fold, you will find model checkpoints, training logs, and final test scores.

## Testing a trained experiment
To evaluate a previously trained cross‑validation run (e.g., after modifying the model or for analysis):
The model will output the classification results for each fold, including metrics such as accuracy and loss.

```bash
python test_crossval.py results/EXPERIMENT_DIR
```
This produces the predictions `test_probs_*.csv` for **submission** to the scoreboard.

## Single‑fold experimentation
For quick experiments on one fold (e.g., tuning hyperparameters), use the standalone training script:
```bash
python train.py --fold 1 --lr 0.01 --epochs 1
```
This does not run the full cross‑validation but is useful for prototyping and debugging.