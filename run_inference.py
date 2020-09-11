import argparse
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config_parser import get_config
from env_config import DATA_PATH
from utils import logger, save_catalog
from data import COLUMNS_KIDS_ALL, COLUMNS_SDSS, process_kids
from models import get_model, build_outputs, get_single_problem_predictions, build_ann_validation_data, \
    build_xgb_validation_data

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config_file', required=True, help='config file name')
parser.add_argument('-r', '--read', dest='read', action='store_true', help='flag to read weights instead of training')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for catalog saving')
parser.add_argument('-t', '--tag', dest='tag', help='catalog tag, added to logs name')
parser.add_argument('--test', dest='is_test', action='store_true', help='indicate test run')
args = parser.parse_args()

cfg = get_config(args, is_inference=True)

# Paths to read weights if args.read is set
if cfg['pred_class']:
    weights_path = 'outputs/inf_models/KiDS_DR4_x_SDSS_DR14_ann_clf_f-all__2020-06-08_15:22:15.hdf5'
else:
    weights_path = 'outputs/inf_models/KiDS_DR4_x_SDSS_DR14_ann_z_f-all_spec-qso__2020-06-08_16:22:38.hdf5'

# Limit rows to read from data in case of a test run
n_rows = 4000 if cfg['is_test'] else None

# Define data paths
data_path_train = path.join(DATA_PATH, 'KiDS/DR4/{train_data}.fits'.format(train_data=cfg['train_data']))
data_path_pred = path.join(DATA_PATH, 'KiDS/DR4/{inference_data}.fits'.format(inference_data=cfg['inference_data']))

# Read train data
logger.info('Reading train data..')
train_data = process_kids(data_path_train, columns=COLUMNS_KIDS_ALL+COLUMNS_SDSS, bands=cfg['bands'], cut=cfg['cut'],
                          sdss_cleaning=True, update_kids=True)

# Limit train data to a given class in case of redshift prediction
if cfg['specialization']:
    mask = (train_data['CLASS'] == cfg['specialization'])
    train_data = train_data.loc[mask]

X = train_data[cfg['features']].values
y = train_data['CLASS'].values
z = train_data['Z'].values

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

# Print information about available observations in each class
classes = np.unique(y)
logger.info('Available classes: {}'.format(np.unique(y, return_counts=True)))

# Train test split
X_train, X_test, y_train, y_test, z_train, z_test, idx_train, idx_test = train_test_split(
    X, y_encoded, z, train_data.index, test_size=0.1, random_state=427)

model = get_model(cfg)
if args.read:
    # Read already trained weights
    logger.info('Loading weights..')
    # TODO: save scaler or the whole pipeline
    model.scaler.fit_transform(X_train)
    model.network = model.create_network(model.params_exp)
    model.network.load_weights(weights_path)

else:
    # Create all the train parameters
    # TODO: refactor, the same in run_experiment, build_validation_data function
    train_params = {}
    if cfg['model'] == 'ann':
        test_names = ['random']
        train_params['validation_data_arr'] = build_ann_validation_data([X_test], [y_test], [z_test], test_names, cfg)
    elif cfg['model'] == 'xgb':
        train_params['eval_set'] = build_xgb_validation_data([X_test], [y_test], [z_test], cfg)

    # Train the model
    logger.info('Training the model..')
    true_outputs = build_outputs(y_train, z_train, cfg)
    model.fit(X_train, true_outputs, **train_params)

# Read inference data
logger.info('Reading inference data..')
data = process_kids(data_path_pred, columns=COLUMNS_KIDS_ALL, bands=cfg['bands'], cut=cfg['cut'], sdss_cleaning=False,
                    n=n_rows, update_kids=True)
X = data[cfg['features']].values

# Predict on the inference data
logger.info('Predicting..')
if cfg['model'] == 'ann':
    preds = model.predict(X, encoder, batch_size=1024)
else:
    preds = get_single_problem_predictions(model, X, encoder, cfg)

# Store predictions
catalog_df = pd.concat([data[['ID', 'RAJ2000', 'DECJ2000', 'MAG_GAAP_r', 'CLASS_STAR', 'MASK']], preds], axis=1)
# TODO: Save information on train/validation split
catalog_df['is_train'] = catalog_df['ID'].isin(train_data['ID'])
if args.save:
    save_catalog(catalog_df, exp_name=cfg['exp_name'], timestamp=cfg['timestamp_start'])
