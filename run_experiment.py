import datetime
import argparse
from config_parser import parse_config
from functools import partial

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils import logger, process_kids, print_feature_ranking, MAG_GAAP_CALIB_R

timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for predictions saving')
args = parser.parse_args()

model, cfg = parse_config(args.config)

# Define validation metrics
metrics = {
    'accuracy': accuracy_score,
    'f1': partial(f1_score, average=None),
}

# Read data
data_path = '/media/snakoneczny/data/KiDS/{data_name}.cols.csv'.format(data_name=cfg['data_name'])
data = process_kids(data_path, sdss_cleaning=True, cut=cfg['cut'])

# Create X and y
X = data[cfg['features']]
y = data['CLASS']

classes = np.unique(y)
logger.info('Available classes: {}'.format(np.unique(y, return_counts=True)))


# Train test split
X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(X, y, X.index, test_size=0.2,
                                                                                     random_state=427)

# Store all predictions and scores
predictions_df = data.loc[idx_train_val, ['ID', 'CLASS']].reset_index(drop=True)
scores = {metric_name: [] for metric_name in metrics}

# Cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds)
for fold, (train, val) in enumerate(kf.split(X_train_val)):
    X_train, y_train = X_train_val.iloc[train], y_train_val.iloc[train]
    X_val, y_val = X_train_val.iloc[val], y_train_val.iloc[val]
    logger.info('fold {}/{}, train: {}, validation: {}'.format(fold + 1, n_folds, train.shape, val.shape))

    # Train a model
    model.fit(X_train, y_train)

    # Evaluate a fold
    y_pred_proba_val = model.predict_proba(X_val)
    y_pred_val = [classes[np.argmax(proba)] for proba in y_pred_proba_val]
    # y_pred_test = clf.predict(X_test)

    # Store scores
    for metric_name, metric_func in metrics.items():
        score = np.around(metric_func(y_val, y_pred_val), 4)
        scores[metric_name].append(score)

    # Store prediction in original data order
    for i, c in enumerate(classes):
        predictions_df.loc[val, c] = y_pred_proba_val[:, i]
    predictions_df.loc[val, 'fold'] = fold

# Report scores
for metric_name in metrics:
    mean_values = np.around(np.mean(scores[metric_name], axis=0), 4)
    std_values = np.around(np.std(scores[metric_name], axis=0), 4)

    logger.info('Validation {metric_name}: {mean} +/- {std}, values: {values}'.format(
        metric_name=metric_name, values=scores[metric_name], mean=mean_values, std=std_values))

# Test
model.fit(X_train_val, y_train_val)

y_pred_proba_test = model.predict_proba(X_test)
y_pred_test = [classes[np.argmax(proba)] for proba in y_pred_proba_test]

for metric_name, metric_func in metrics.items():
    score = np.around(metric_func(y_test, y_pred_test), 4)
    logger.info('Test {metric_name}: {score}'.format(metric_name=metric_name, score=score))

print_feature_ranking(model, X_train_val)

# Save predictions df
if args.save:
    predictions_path = 'experiments/{exp_name}__{timestamp}.csv'.format(exp_name=cfg['exp_name'],
                                                                        timestamp=timestamp_start)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info('predictions saved to: {}'.format(predictions_path))
