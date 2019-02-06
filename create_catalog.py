import datetime
import argparse
from config_parser import parse_config

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import logger
from data import BAND_CALIB_COLUMNS, COLOR_COLUMNS, process_kids, process_kids_data

COLUMNS_TO_ADD = ['RAJ2000', 'DECJ2000', 'CLASS_STAR', BAND_CALIB_COLUMNS, COLOR_COLUMNS]


def create_catalog_chunk(data_chunk, y_pred_proba, classes):
    y_pred = [classes[np.argmax(proba)] for proba in y_pred_proba]

    catalog_chunk = pd.DataFrame()
    catalog_chunk['ID'] = data_chunk['ID']
    catalog_chunk['CLASS'] = y_pred

    for i, c in enumerate(classes):
        catalog_chunk[c] = y_pred_proba[:, i]

    for columns in COLUMNS_TO_ADD:
        catalog_chunk[columns] = data_chunk[columns]

    return catalog_chunk


timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', required=True, help='config file name')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='flag for catalog saving')
args = parser.parse_args()

model_constructor, cfg = parse_config(args.config)
model = model_constructor(cfg)

# Create data paths
data_path_train = '/media/snakoneczny/data/KiDS/{train_data}.cols.csv'.format(train_data=cfg['train_data'])
data_path_pred = '/media/snakoneczny/data/KiDS/{inference_data}.cols.csv'.format(inference_data=cfg['inference_data'])

# Read and process train data
data = process_kids(data_path_train, sdss_cleaning=cfg['clean_sdss'], cut=cfg['cut'])

# Create X and y
X = data[cfg['features']]
y = data['CLASS']

classes = np.unique(y)
logger.info('training classes: {}'.format(np.unique(y, return_counts=True)))

# Train a model
model.fit(X, y)

# Process chunks of catalog data
catalog_df = pd.DataFrame()

for data_chunk in tqdm(pd.read_csv(data_path_pred, chunksize=4000000), desc='Inference'):
    data_chunk = process_kids_data(data_chunk, sdss_cleaning=False, cut=cfg['cut'], with_print=False)

    X = data_chunk[cfg['features']]
    y_pred_proba = model.predict_proba(X)

    catalog_chunk = create_catalog_chunk(data_chunk, y_pred_proba, classes)
    catalog_df = catalog_df.append(catalog_chunk)

logger.info('catalog size: {}'.format(catalog_df.shape))

# TODO: make sure galactic coordinates are being saved as well
# Save catalog
if args.save:
    catalog_path = 'outputs/catalogs/{exp_name}__{timestamp}.csv'.format(exp_name=cfg['exp_name'],
                                                                         timestamp=timestamp_start)
    catalog_df.to_csv(catalog_path, index=False)
    logger.info('catalog saved to: {}'.format(catalog_path))
