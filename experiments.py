from itertools import chain
from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

from utils import logger, safe_indexing
from evaluation import metric_class_split
from models import get_single_predictions

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


def do_experiment(data, model, cfg, encoder, X_train, X_test, y_train, y_test, z_train, z_test, idx_test):
    predictions_df = data.loc[idx_test, ['ID', 'CLASS', 'Z']].reset_index(drop=True)

    # Train the model
    true_outputs = build_outputs(y_train, z_train, cfg)
    # TODO: cos z is_validation
    train_params = {}
    if cfg['model'] == 'ann':
        train_params['validation_data'] = build_validation_data(X_test, y_test, z_test, cfg)
    model.fit(X_train, true_outputs, **train_params)

    # Predict on the validation data
    if cfg['model'] == 'ann':
        preds_val = model.predict(X_test, encoder)
    else:
        preds_val = get_single_predictions(model, X_test, encoder, cfg)

    # Store prediction in original data order
    predictions_df = pd.concat([predictions_df, preds_val], ignore_index=True, axis=1)
    predictions_df['subset'] = 'test'

    # Get and store scores
    scores, report = get_scores(y_test, z_test, preds_val, encoder, cfg)

    return predictions_df, scores, report


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
        predictions_df.loc[val, 'subset'] = 'fold {}'.format(fold)

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


def top_k_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')
    test_size = options.pop('test_size', 'default')

    if options:
        raise TypeError('Invalid parameters passed: %s' % str(options))

    if test_size == 'default':
        test_size = 0.1

    # Make indexable
    arrays = [a for a in arrays]

    # Get top and low index
    k = int(test_size * arrays[0].shape[0])
    ind_part = np.argpartition(arrays[0], -1 * k)
    ind_top = ind_part[-1 * k:]
    ind_low = ind_part[:-1 * k]

    return list(chain.from_iterable((safe_indexing(a, ind_low), safe_indexing(a, ind_top)) for a in arrays))


def build_ann_output_dict(y, z, cfg):
    outputs = {}
    if cfg['pred_class']:
        outputs['category_output'] = y
    if cfg['pred_z']:
        outputs['redshift_output'] = z
    return outputs


def build_outputs(y, z, cfg):
    if cfg['model'] == 'ann':
        outputs = build_ann_output_dict(y, z, cfg)
    else:
        if cfg['pred_class']:
            outputs = y
        else:
            outputs = z
    return outputs


def build_validation_data(X_val, y_val, z_val, cfg):
    validation_outputs = build_ann_output_dict(y_val, z_val, cfg)
    validation_data = (X_val, validation_outputs)
    return validation_data
