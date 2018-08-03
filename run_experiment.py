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
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

from utils import logger, process_kids, print_feature_ranking
from utils_evaluation import metric_class_split

timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for predictions saving')
args = parser.parse_args()

# TODO
model_constructor, cfg = parse_config(args.config)
model = model_constructor(cfg)

# Define validation metrics
metrics_classification = OrderedDict([
    ('accuracy', accuracy_score),
    ('f1', partial(f1_score, average=None))
])

metrics_redshift = OrderedDict([
    ('mean_square_error', mean_squared_error),
    ('mean_absolute_error', mean_absolute_error),
])

metrics_redshift_per_class = OrderedDict([
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

# TODO
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
encoded_y = np_utils.to_categorical(encoded_y)

# Print information about available observations in each class
classes = np.unique(y)
logger.info('Available classes: {}'.format(np.unique(y, return_counts=True)))

# Train test split
X_train_val, X_test, y_train_val, y_test, z_train_val, z_test, idx_train_val, idx_test = train_test_split(
    X, encoded_y, z, data.index, test_size=0.2, random_state=427)

# Store all predictions and scores
predictions_df = data.loc[idx_train_val, ['ID', 'CLASS', 'Z']].reset_index(drop=True)
scores = {metric_name: [] for metric_name in
          {**metrics_classification, **metrics_redshift, **metrics_redshift_per_class}}

# Cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds)
for fold, (train, val) in enumerate(kf.split(X_train_val)):
    X_train, y_train, z_train = X_train_val[train], y_train_val[train], z_train_val[train]
    X_val, y_val, z_val = X_train_val[val], y_train_val[val], z_train_val[val]
    logger.info('fold {}/{}, train: {}, validation: {}'.format(fold + 1, n_folds, train.shape, val.shape))

    # TODO
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # TODO
    # Train a model
    # model.fit(X_train, y_train)
    model.fit(X_train, {'category_output': y_train, 'redshift_output': z_train},
              validation_data=(X_val, {'category_output': y_val, 'redshift_output': z_val}),
              epochs=2, batch_size=32, verbose=1)

    # Evaluate a fold
    preds_val = model.predict(X_val)
    y_pred_proba_val = preds_val[0]
    z_pred_val = preds_val[1]
    # y_pred_proba_val = model.predict(X_val)[0]  # TODO: proba
    # y_pred_val = [classes[np.argmax(proba)] for proba in y_pred_proba_val]

    y_pred_val_decoded = encoder.inverse_transform(np.argmax(y_pred_proba_val, axis=1))
    y_val_decoded = encoder.inverse_transform(np.argmax(y_val, axis=1))

    # Store scores  # TODO: refactor
    for metric_name, metric_func in metrics_classification.items():
        score = np.around(metric_func(y_val_decoded, y_pred_val_decoded), 4)
        scores[metric_name].append(score)

    for metric_name, metric_func in metrics_redshift.items():
        score = np.around(metric_func(z_val, z_pred_val), 4)
        scores[metric_name].append(score)

    for metric_name, metric_func in metrics_redshift_per_class.items():
        score = np.around(metric_func(z_val, z_pred_val, y_val_decoded), 4)
        scores[metric_name].append(score)

    # Store prediction in original data order
    for i, c in enumerate(encoder.classes_):
        predictions_df.loc[val, c] = y_pred_proba_val[:, i]
    predictions_df.loc[val, 'Z_pred'] = z_pred_val

    # Store fold number
    predictions_df.loc[val, 'fold'] = fold

# Report validation scores
report_lines = ['\n']
for metric_name in OrderedDict(chain(metrics_classification.items(), metrics_redshift.items(), metrics_redshift_per_class.items())):
    mean_values = np.around(np.mean(scores[metric_name], axis=0), 4)
    std_values = np.around(np.std(scores[metric_name], axis=0), 4)

    report_lines.append('Validation {metric_name}: {mean} +/- {std}, values: {values}'.format(
        metric_name=metric_name, values=scores[metric_name], mean=mean_values, std=std_values))

logger.info('\n'.join(report_lines))

# Train test model on all training data  # TODO
scaler = MinMaxScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

# TODO
model.fit(X_train_val, {'category_output': y_train_val, 'redshift_output': z_train_val})

# Predict on test
# TODO: proba (?)
preds_test = model.predict(X_test)
y_pred_proba_test = preds_test[0]
z_pred_test = preds_test[1]
# y_pred_test = [classes[np.argmax(proba)] for proba in y_pred_proba_test]

y_pred_test_decoded = encoder.inverse_transform(np.argmax(y_pred_proba_test, axis=1))
y_test_decoded = encoder.inverse_transform(np.argmax(y_test, axis=1))

# Report test scores
report_Lines = ['\n']
for metric_name, metric_func in metrics_classification.items():
    score = np.around(metric_func(y_test_decoded, y_pred_test_decoded), 4)
    report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

for metric_name, metric_func in metrics_redshift.items():
    score = np.around(metric_func(z_test, z_pred_test), 4)
    report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

for metric_name, metric_func in metrics_redshift_per_class.items():
    score = np.around(metric_func(z_test, z_pred_test, y_test_decoded), 4)
    report_lines.append('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

logger.info('\n'.join(report_lines))

# Print feature ranking
if cfg['model'] in ['rf', 'xgb']:
    print_feature_ranking(model, X_train_val)

# Save predictions df
if args.save:
    predictions_path = 'experiments/{exp_name}__{timestamp}.csv'.format(exp_name=cfg['exp_name'],
                                                                        timestamp=timestamp_start)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info('predictions saved to: {}'.format(predictions_path))
