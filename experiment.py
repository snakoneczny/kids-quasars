from itertools import chain
from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

from utils import logger, safe_indexing
from evaluation import metric_class_split
from models import get_single_problem_predictions, build_outputs, build_ann_validation_data, build_xgb_validation_data

# Define validation metrics
metrics_classification = OrderedDict([
    ('accuracy', accuracy_score),
    ('f1', partial(f1_score, average=None))
])

metrics_redshift = OrderedDict([
    ('mean_square_error', mean_squared_error),
    ('mean_absolute_error', mean_absolute_error),
    ('mse_classes', partial(metric_class_split, metric=mean_squared_error)),
    ('mae_classes', partial(metric_class_split, metric=mean_absolute_error)),
])


def do_experiment(data, model, cfg, encoder, X_train, X_test_arr, y_train, y_test_arr, z_train, z_test_arr,
                  idx_test_arr, test_names_arr):
    # Get the data frame with all objects used as test
    idx_test = np.concatenate(idx_test_arr)
    predictions_df = data.loc[idx_test, ['ID', 'CLASS', 'Z']].reset_index(drop=True)

    # Limit train sample and test samples on which scores are calculated to the specialized subset
    X_train_nospec, X_test_arr_nospec, y_train_nospec, y_test_arr_nospec, z_train_nospec, z_test_arr_nospec = \
        X_train, X_test_arr, y_train, y_test_arr, z_train, z_test_arr
    if cfg['specialization']:
        X_train, X_test_arr, y_train, y_test_arr, z_train, z_test_arr = \
            limit_to_spec(cfg, encoder, X_train, X_test_arr, y_train, y_test_arr, z_train, z_test_arr)

    # Create all the train parameters
    # TODO: refactor, build_validation_data function
    true_outputs = build_outputs(y_train, z_train, cfg)
    train_params = {}
    if cfg['model'] == 'ann':
        train_params['validation_data_arr'] = build_ann_validation_data(X_test_arr, y_test_arr, z_test_arr,
                                                                        test_names_arr, cfg)
    elif cfg['model'] == 'xgb':
        train_params['eval_set'] = build_xgb_validation_data(X_test_arr, y_test_arr, z_test_arr, cfg)
        train_params['early_stopping_rounds'] = 200  # TODO: extract some train parameters

    # Train the model
    model.fit(X_train, true_outputs, **train_params)

    # Predict on the validation data
    if cfg['model'] == 'ann':
        preds_val_arr = [model.predict(X_test_nospec, encoder) for X_test_nospec in X_test_arr_nospec]
    else:
        preds_val_arr = [get_single_problem_predictions(model, X_test_nospec, encoder, cfg) for X_test_nospec in
                         X_test_arr_nospec]

    # Fill test name, concat rows wise, concat with original data
    for i, test_name in enumerate(test_names_arr):
        preds_val_arr[i]['test_subset'] = test_name
    preds_val = pd.concat(preds_val_arr, axis=0).reset_index(drop=True)
    predictions_df = pd.concat([predictions_df, preds_val], axis=1)

    # Get and store scores
    if cfg['specialization']:
        # Limit validation predictions to specialized subset when calculating scores
        mask = (y_test_arr_nospec[0] == encoder.transform([cfg['specialization']])[0])
        preds_val_arr[0] = preds_val_arr[0][mask]

    scores, report = get_scores(y_test_arr[0], z_test_arr[0], preds_val_arr[0], encoder, cfg)

    return predictions_df, scores, report


def limit_to_spec(cfg, encoder, X_train, X_test_arr, y_train, y_test_arr, z_train, z_test_arr):
    # Limit train data
    spec_encoded = encoder.transform([cfg['specialization']])[0]
    mask = (y_train == spec_encoded)
    X_train_spec, y_train_spec, z_train_spec = X_train[mask], y_train[mask], z_train[mask]

    # Limit test data in arrays
    X_test_arr_spec, y_test_arr_spec, z_test_arr_spec = X_test_arr.copy(), y_test_arr.copy(), z_test_arr.copy()
    for i in range(len(X_test_arr)):
        mask = (y_test_arr[i] == spec_encoded)
        X_test_arr_spec[i], y_test_arr_spec[i], z_test_arr_spec[i] = \
            X_test_arr[i][mask], y_test_arr[i][mask], z_test_arr[i][mask]

    return X_train_spec, X_test_arr_spec, y_train_spec, y_test_arr_spec, z_train_spec, z_test_arr_spec


def kfold_validation(data, model, cfg, encoder, X_train_val, y_train_val, z_train_val, idx_train_val):
    # Store all predictions and scores
    predictions_df = data.loc[idx_train_val, ['ID', 'CLASS', 'Z']].reset_index(drop=True)
    scores = {metric_name: [] for metric_name in {**metrics_classification, **metrics_redshift}}
    # Cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds)
    for fold, (train, val) in enumerate(kf.split(X_train_val)):
        X_train, y_train, z_train = X_train_val[train], y_train_val[train], z_train_val[train]
        X_val, y_val, z_val = X_train_val[val], y_train_val[val], z_train_val[val]
        logger.info('fold {}/{}, train: {}, validation: {}'.format(fold + 1, n_folds, train.shape, val.shape))

        preds_val, scores_val, _ = do_experiment(data, model, cfg, encoder, X_train, X_val, y_train, y_val, z_train,
                                                 z_val, val)

        # Add fold scores from dictionary to scores stored in dictionary of arrays
        for metric_name in scores_val:
            scores[metric_name].append(scores_val[metric_name])

        # Store prediction in original data order
        predictions_df.loc[val] = preds_val
        predictions_df.loc[val, 'test'] = 'fold {}'.format(fold)

    # Report kfold validation scores
    report = get_kfold_report(scores)

    return predictions_df, report


def get_kfold_report(scores):
    report_lines = []
    for metric_name in OrderedDict(chain(metrics_classification.items(), metrics_redshift.items())):

        if len(scores[metric_name]) > 0:
            mean_values = np.around(np.mean(scores[metric_name], axis=0), 4)
            std_values = np.around(np.std(scores[metric_name], axis=0), 4)

            report_lines.append('Validation {metric_name}: {mean} +/- {std}, values: {values}'.format(
                metric_name=metric_name, values=scores[metric_name], mean=mean_values, std=std_values))

    return '\n'.join(report_lines)


def get_scores(y_val, z_val, preds_val, encoder, cfg):
    scores = {}
    report_lines = []

    # Get and store classification scores
    y_val_decoded = encoder.inverse_transform(y_val)
    if cfg['pred_class']:  # or CLASS_PHOTO in preds
        for metric_name, metric_func in metrics_classification.items():
            score = np.around(metric_func(y_val_decoded, preds_val['CLASS_PHOTO']), 4)
            scores[metric_name] = score
            # TODO: it works but it assumes "Test"
            report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

    # Get and store regression scores
    if cfg['pred_z']:
        for metric_name, metric_func in metrics_redshift.items():
            if metric_name in ['mse_classes', 'mae_classes']:
                metric_func = partial(metric_func, classes=y_val_decoded)
            score = np.around(metric_func(z_val, preds_val['Z_PHOTO']), 4)
            scores[metric_name] = score
            report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

    return scores, '\n'.join(report_lines)


def train_test_top_split(*arrays, test_size=0.2):
    """
    :param arrays: arrays of any format, the first one should be array or Series
        All arrays are divided based on the values in the first one
    :param test_size: float (0, 1)
    :return: list
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    # Make indexable
    arrays = [a for a in arrays]

    # Get top and low index
    k = int(test_size * arrays[0].shape[0])
    ind_part = np.argpartition(arrays[0], -1 * k)
    ind_top = ind_part[-1 * k:]
    ind_low = ind_part[:-1 * k]

    return list(chain.from_iterable((safe_indexing(a, ind_low), safe_indexing(a, ind_top)) for a in arrays))


def train_test_top_random_split(*arrays, top_test_size=0.1, random_test_size=0.1):
    """
    :param arrays: arrays of any format, the first one should be array or Series
        All arrays are divided based on the values in the first one
    :param top_test_size: float (0, 1)
    :param random_test_size: float (0, 1)
    :return: list
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    # Make indexable
    arrays = [a for a in arrays]

    # Get top index
    k = int(top_test_size * arrays[0].shape[0])
    split_ind = arrays[0].shape[0] - k
    ind_part = np.argpartition(arrays[0], split_ind)
    ind_test_top = ind_part[split_ind:]
    ind_low = ind_part[:split_ind]

    ind_train, ind_test_random = train_test_split(ind_low, test_size=random_test_size, random_state=8725)

    return list(chain.from_iterable((safe_indexing(a, ind_train), safe_indexing(a, ind_test_top),
                                     safe_indexing(a, ind_test_random)) for a in arrays))


def value_split(*arrays, value):
    """
    :param arrays: arrays of any format, the first one should be array or Series
        All arrays are divided based on the values in the first one
    :param value: float
    :return: list
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    # Make indexable
    arrays = [a for a in arrays]

    # Get top and low index
    ind_top = np.where(arrays[0] >= value)[0]
    ind_low = np.where(arrays[0] < value)[0]

    return list(chain.from_iterable((safe_indexing(a, ind_low), safe_indexing(a, ind_top)) for a in arrays))
