"""Disjoint module splits. No observation-level random splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def validate_index(index):
    if not {'file_name', 'Label', 'folder_index'} <= set(index):
        raise ValueError('Index needs file_name, Label, folder_index')
    if index.file_name.duplicated().any() or index.file_name.isna().any():
        raise ValueError('Each module must occur exactly once in the index')
    for name in index.file_name:
        if not isinstance(name, str) or '/' in name or '\\' in name or Path(name).name != name or not name.endswith('.csv'):
            raise ValueError(f'Expected a plain module CSV basename: {name!r}')
    if not set(index.Label.unique()) <= {0, 1}:
        raise ValueError('Labels must be binary')
    if not set(index.folder_index.unique()) <= {1, 2, 3}:
        raise ValueError('Legacy folder_index must be 1, 2 or 3')


def fixed_manifest(index):
    """80:20 stratified holdout, seed 42; three inner validation folds."""
    validate_index(index)
    frame = index.sort_values('file_name').reset_index(drop=True).copy()
    train_ids, test_ids = train_test_split(frame.index.to_numpy(), test_size=.2,
                                         stratify=frame.Label, random_state=42)
    frame['subset'] = 'training'
    frame.loc[test_ids, 'subset'] = 'test'
    frame['cv_fold'] = 0
    train = frame.loc[frame.subset == 'training']
    for fold, (_, val_ids) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(train, train.Label), 1):
        frame.loc[train.index[val_ids], 'cv_fold'] = fold
    return frame


def inner_partitions(manifest, fold):
    train = manifest.subset.eq('training')
    return dict(train=manifest.loc[train & manifest.cv_fold.ne(fold), 'file_name'].tolist(),
                validation=manifest.loc[train & manifest.cv_fold.eq(fold), 'file_name'].tolist(),
                test=[])
