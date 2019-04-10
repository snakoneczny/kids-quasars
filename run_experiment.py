import datetime
import argparse
from config_parser import parse_config

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import logger, save_predictions, save_model
from data import get_mag_str, process_kids
from experiments import kfold_validation, top_k_split, do_experiment
from plotting import plot_feature_ranking


timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for predictions saving')
args = parser.parse_args()

model_constructor, cfg = parse_config(args.config)
model = model_constructor(cfg)

# Read data
data_path = '/media/snakoneczny/data/KiDS/DR4/{train_data}.fits'.format(train_data=cfg['train_data'])
data = process_kids(data_path, bands=cfg['bands'], cut=cfg['cut'], sdss_cleaning=True,)

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

if cfg['test'] == 'kfold':

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


elif cfg['test'] == 'magnitude':

    # Train test split
    _, _, X_train, X_test, y_train, y_test, z_train, z_test, idx_train, idx_test = \
        top_k_split(data[get_mag_str('r')], X, y_encoded, z, data.index, test_size=0.1)

    # Testing
    predictions, scores, report = do_experiment(
        data, model, cfg, encoder, X_train, X_test, y_train, y_test, z_train, z_test, idx_test)

    # Finish by showing reports
    logger.info(report)


elif cfg['test'] == 'redshift':
    raise Exception('Top redshift testing not implemented')

else:
    raise Exception('Unknown test method: {}'.format(cfg['test']))

# Plot feature importance of the last model
plot_feature_ranking(model, cfg['features'])

if args.save:
    save_predictions(predictions, timestamp_start, cfg)
    save_model(model, timestamp_start, cfg)
