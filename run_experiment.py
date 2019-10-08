import argparse
from os import path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config_parser import get_config
from env_config import DATA_PATH
from utils import logger, save_predictions, save_model
from data import get_mag_str, process_kids
from experiments import kfold_validation, top_k_split, do_experiment
from plotting import plot_feature_ranking
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_file', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for predictions saving')
parser.add_argument('-t', '--tag', dest='tag', help='experiment tag, added to logs name')
parser.add_argument('--test', dest='test', action='store_true', help='indicate test run')
args = parser.parse_args()

cfg = get_config(args)
model = get_model(cfg)

# Read data
data_path = path.join(DATA_PATH, 'KiDS/DR4/{train_data}.fits'.format(train_data=cfg['train_data']))
data = process_kids(data_path, bands=cfg['bands'], cut=cfg['cut'], sdss_cleaning=True)

# Get X and y
X = data[cfg['features']].values
y = data['CLASS'].values
z = data['Z'].values

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

# Print information about available observations in each class
classes = np.unique(y)
logger.info('Available classes: {}'.format(np.unique(y, return_counts=True)))

if cfg['test_method'] == 'kfold':

    # Train test split
    X_train_val, X_test, y_train_val, y_test, z_train_val, z_test, idx_train_val, idx_test = train_test_split(
        X, y_encoded, z, data.index, test_size=0.2, random_state=427)

    # Validation
    predictions_val, validation_report = kfold_validation(data, model, cfg, encoder,
                                                          X_train_val, y_train_val, z_train_val, idx_train_val)

    # Testing
    predictions_test, scores_test, test_report = do_experiment(
        data, model, cfg, encoder, X_train_val, X_test, y_train_val, y_test, z_train_val, z_test, idx_test)

    predictions = pd.concat([predictions_val, predictions_test])

    # Finish by showing reports
    logger.info(validation_report)
    logger.info(test_report)

elif cfg['test_method'] == 'random':
    # Train test split
    X_train, X_test, y_train, y_test, z_train, z_test, idx_train, idx_test = train_test_split(
        X, y_encoded, z, data.index, test_size=0.2, random_state=427)

    # Testing
    predictions, scores, report = do_experiment(
        data, model, cfg, encoder, X_train, X_test, y_train, y_test, z_train, z_test, idx_test)

    # Finish by showing reports
    logger.info(report)

elif cfg['test_method'] == 'magnitude':
    # Train test split
    _, _, X_train, X_test, y_train, y_test, z_train, z_test, idx_train, idx_test = \
        top_k_split(data[get_mag_str('r')], X, y_encoded, z, data.index, test_size=0.1)

    # Testing
    predictions, scores, report = do_experiment(
        data, model, cfg, encoder, X_train, X_test, y_train, y_test, z_train, z_test, idx_test)

    # Finish by showing reports
    logger.info(report)


elif cfg['test_method'] == 'redshift':
    raise Exception('Top redshift testing not implemented')

else:
    raise Exception('Unknown test method: {}'.format(cfg['test']))

# Plot feature importance of the last model
if cfg['model'] == 'rf':
    plot_feature_ranking(model, cfg['features'])

if args.save:
    save_predictions(predictions, exp_name=cfg['exp_name'], timestamp=cfg['timestamp_start'])

    # Models are needed only for feature importance, accessible only in tree methods
    if cfg['model'] != 'ann':
        save_model(model, exp_name=cfg['exp_name'], timestamp=cfg['timestamp_start'])
