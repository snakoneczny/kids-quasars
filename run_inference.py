import argparse
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config_parser import get_config
from env_config import DATA_PATH
from utils import logger, save_catalog
from data import process_kids
from models import get_model, build_outputs, get_single_problem_predictions

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_file', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for catalog saving')
parser.add_argument('-t', '--tag', dest='tag', help='catalog tag, added to logs name')
parser.add_argument('--test', dest='is_test', action='store_true', help='indicate test run')
args = parser.parse_args()

cfg = get_config(args, is_inference=True)

# Limit rows to read from data in case of a test run
n_rows = 1000 if cfg['is_test'] else None

# Define data paths
data_path_train = path.join(DATA_PATH, 'KiDS/DR4/{train_data}.fits'.format(train_data=cfg['train_data']))
data_path_pred = path.join(DATA_PATH, 'KiDS/DR4/{inference_data}.fits'.format(inference_data=cfg['inference_data']))

# Read train data
logger.info('Reading train data..')
data = process_kids(data_path_train, bands=cfg['bands'], cut=cfg['cut'], sdss_cleaning=True)

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

# Limit train data to a given class in case of redshift prediction
if cfg['specialization']:
    mask = (y == cfg['specialization'])
    X, y, z = X[mask], y[mask], z[mask]

# Train the model
logger.info('Training model..')
model = get_model(cfg)
true_outputs = build_outputs(y, z, cfg)
model.fit(X, true_outputs)

# Read inference data
logger.info('Reading inference data..')
data = process_kids(data_path_pred, bands=cfg['bands'], cut=cfg['cut'], sdss_cleaning=False, n=n_rows)
X = data[cfg['features']].values

# Predict on the inference data
logger.info('Predicting..')
if cfg['model'] == 'ann':
    preds = model.predict(X, encoder)
else:
    preds = get_single_problem_predictions(model, X, encoder, cfg)

# Store predictions
catalog_df = pd.concat([data['ID'], preds], axis=1)
if args.save:
    save_catalog(catalog_df, exp_name=cfg['exp_name'], timestamp=cfg['timestamp_start'])
