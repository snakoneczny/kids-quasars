import datetime
import argparse
from config_parser import parse_config
from functools import partial
from collections import OrderedDict
from itertools import chain

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from utils import logger, process_kids, print_feature_ranking, get_estimation_type, get_model_type
from utils_evaluation import metric_class_split

timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for predictions saving')
args = parser.parse_args()

model_constructor, cfg = parse_config(args.config)
model = model_constructor(cfg)
estimation_type = get_estimation_type(cfg['model'])
model_type = get_model_type(cfg['model'])

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

# Read data
data_path = '/media/snakoneczny/data/KiDS/{data_name}.cols.csv'.format(data_name=cfg['data_name'])
data = process_kids(data_path, subset=cfg['subset'], sdss_cleaning=cfg['clean_sdss'], cut=cfg['cut'])

# Get X and y
X = data[cfg['features']].as_matrix()
y = data['CLASS'].as_matrix()
z = data['Z'].as_matrix()

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

# Print information about available observations in each class
classes = np.unique(y)
logger.info('Available classes: {}'.format(np.unique(y, return_counts=True)))

# Train test split
X_train_val, X_test, y_train_val, y_test, z_train_val, z_test, idx_train_val, idx_test = train_test_split(
    X, y_encoded, z, data.index, test_size=0.2, random_state=427)


def validation(data, X_train_val, y_train_val, z_train_val, idx_train_val):
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

        # Train a model
        if estimation_type == 'astronet':
            model.fit(X_train, {'category_output': y_train, 'redshift_output': z_train},
                      validation_data=(X_val, {'category_output': y_val, 'redshift_output': z_val}))
        elif estimation_type == 'clf':
            model.fit(X_train, y_train)
        else:
            assert estimation_type == 'reg'
            model.fit(X_train, z_train)

        # Predict on validation data
        if estimation_type == 'astronet':
            preds_val = model.predict(X_val)
            y_pred_proba_val = preds_val[0]
            z_pred_val = preds_val[1]
        elif estimation_type == 'clf':
            y_pred_proba_val = model.predict_proba(X_val)
        else:  # model_type == 'reg':
            z_pred_val = model.predict(X_val)

        # Get and store classification scores  # TODO: refactor
        y_val_decoded = encoder.inverse_transform(y_val)
        if estimation_type in ['astronet', 'clf']:

            # Store prediction in original data order
            for i, c in enumerate(encoder.classes_):
                predictions_df.loc[val, c] = y_pred_proba_val[:, i]

            y_pred_val_decoded = encoder.inverse_transform(np.argmax(y_pred_proba_val, axis=1))

            for metric_name, metric_func in metrics_classification.items():
                score = np.around(metric_func(y_val_decoded, y_pred_val_decoded), 4)
                scores[metric_name].append(score)

        # Get and store regression scores
        if estimation_type in ['astronet', 'reg']:

            # Store prediction in original data order
            predictions_df.loc[val, 'Z_pred'] = z_pred_val

            for metric_name, metric_func in metrics_redshift.items():
                if metric_name in ['mse_classes', 'mae_classes']:
                    metric_func = partial(metric_func, classes=y_val_decoded)
                score = np.around(metric_func(z_val, z_pred_val), 4)
                scores[metric_name].append(score)

        # Store fold number
        predictions_df.loc[val, 'fold'] = fold

    # Report validation scores
    report_lines = []
    for metric_name in OrderedDict(chain(metrics_classification.items(), metrics_redshift.items())):

        if len(scores[metric_name]) > 0:
            mean_values = np.around(np.mean(scores[metric_name], axis=0), 4)
            std_values = np.around(np.std(scores[metric_name], axis=0), 4)

            report_lines.append('Validation {metric_name}: {mean} +/- {std}, values: {values}'.format(
                metric_name=metric_name, values=scores[metric_name], mean=mean_values, std=std_values))

    return predictions_df, '\n'.join(report_lines)


def test(X_train_val, X_test, y_train_val, y_test, z_train_val, z_test):
    # Train a model
    if estimation_type == 'astronet':
        model.fit(X_train_val, {'category_output': y_train_val, 'redshift_output': z_train_val},
                  validation_data=(X_test, {'category_output': y_test, 'redshift_output': z_test}))
    elif estimation_type == 'clf':
        model.fit(X_train_val, y_train_val)
    else:
        assert estimation_type == 'reg'
        model.fit(X_train_val, z_train_val)

    # Predict on test data
    if estimation_type == 'astronet':
        preds_test = model.predict(X_test)
        y_pred_proba_test = preds_test[0]
        z_pred_test = preds_test[1]
    elif estimation_type == 'clf':
        y_pred_proba_test = model.predict_proba(X_test)
    else:  # model_type == 'reg':
        z_pred_test = model.predict(X_test)

    # Report test scores
    y_test_decoded = encoder.inverse_transform(y_test)
    report_lines = []

    # Classification ones
    if estimation_type in ['astronet', 'clf']:
        y_pred_test_decoded = encoder.inverse_transform(np.argmax(y_pred_proba_test, axis=1))

        for metric_name, metric_func in metrics_classification.items():
            score = np.around(metric_func(y_test_decoded, y_pred_test_decoded), 4)
            report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

    # Regression ones
    if estimation_type in ['astronet', 'reg']:
        for metric_name, metric_func in metrics_redshift.items():
            if metric_name in ['mse_classes', 'mae_classes']:
                metric_func = partial(metric_func, classes=y_test_decoded)
            score = np.around(metric_func(z_test, z_pred_test), 4)
            report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

    return '\n'.join(report_lines)


# Validation
predictions_df, validation_report = validation(data, X_train_val, y_train_val, z_train_val, idx_train_val)

# Testing
test_report = test(X_train_val, X_test, y_train_val, y_test, z_train_val, z_test)

# Print feature ranking
if model_type in ['rf', 'xgb']:
    print_feature_ranking(model, cfg['features'])

# Finish by showing reports
logger.info(validation_report)
logger.info(test_report)

# TODO: right now we save only validation predictions, but test should also be saved if we want to report test scores
# Save predictions df
if args.save:
    predictions_path = 'outputs/experiments/{exp_name}__{timestamp}.csv'.format(exp_name=cfg['exp_name'],
                                                                        timestamp=timestamp_start)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info('predictions saved to: {}'.format(predictions_path))
