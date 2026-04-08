"""
Fixed ESC-50 fold splits – DO NOT MODIFY.
These ensure reproducible cross-validation across all student submissions.
"""

import os
from sklearn.model_selection import train_test_split


def get_esc50_splits(root, test_fold, subset, val_size=0.2, random_state=0):
    """
    Returns file names for requested subset.
    ONE SOURCE OF TRUTH for all split logic.
    """
    # Get all files and split by fold
    all_files = sorted(os.listdir(os.path.join(root, 'ESC-50-master/audio')))
    test_files = [f for f in all_files if int(f.split('-')[0]) == test_fold]
    train_files = [f for f in all_files if int(f.split('-')[0]) != test_fold]

    # Split train into train/val
    train_files, val_files = train_test_split(
        train_files, test_size=val_size, random_state=random_state
    )

    # Return only what's needed
    if subset == "train":
        return train_files
    elif subset == "val":
        return val_files
    elif subset == "test":
        return test_files
    else:
        raise ValueError(f"Unknown subset: {subset}")

# Pre-computed for verification (optional)
EXPECTED_FILE_COUNTS = {
    1: {'train': 1600 - 320, 'val': 320, 'test': 400},  # 80/20 split approx
    2: {'train': 1600 - 320, 'val': 320, 'test': 400},
    3: {'train': 1600 - 320, 'val': 320, 'test': 400},
    4: {'train': 1600 - 320, 'val': 320, 'test': 400},
    5: {'train': 1600 - 320, 'val': 320, 'test': 400},
}