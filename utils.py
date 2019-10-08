import logging
import math
from os import path

import scipy
import numpy as np
import pandas as pd
from scipy import stats
import joblib

from env_config import DATA_PATH

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_indexing(X, indices):
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            # TODO: that was commented
            # warnings.warn("Copying input dataframe for slicing.",
            #               DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def get_map(l, b, nside=128):
    # Set the number of sources and the coordinates for the input
    npix = hp.nside2npix(nside)  # 12 * nside ^ 2

    # Coordinates and the density field f
    phis = l / 180. * math.pi
    thetas = (-1. * b + 90.) / 180. * math.pi

    # Initate the map and fill it with the values
    hpxmap = np.zeros(npix, dtype=np.float)

    # Go from HEALPix coordinates to indices
    indices = hp.ang2pix(nside, thetas, phis, nest=False)
    for i in indices:
        hpxmap[i] += 1

    lon, lat = hp.pixelfunc.pix2ang(nside, range(npix), nest=False, lonlat=True)

    return hpxmap, lon, lat


def get_weighted_map(data_path='maps/2MASS_XSC_full_density_gallactic.csv', nside=128):
    npix = 12 * nside ** 2

    data = pd.read_csv(data_path)

    phis = data['GAL_LONG'] / 180. * math.pi
    thetas = (-1. * data['GAL_LAT'] + 90.) / 180. * math.pi
    indices = hp.ang2pix(nside, thetas, phis, nest=False)

    pixel_sum = [0] * npix
    pixel_w_sum = [0] * npix
    pixel_w_mean = [0] * npix

    for i in indices:
        pixel_sum[i] += 1
        pixel_w_sum[i] += data.loc[i, 'density']

    for i in range(npix):
        if pixel_sum[i] != 0:
            pixel_w_mean[i] = pixel_w_sum[i] / pixel_sum[i]
        else:
            pixel_w_mean[i] = 0
    pixel_w_mean = np.array(pixel_w_mean)

    lon, lat = hp.pixelfunc.pix2ang(nside, range(npix), nest=False, lonlat=True)

    return pixel_w_mean, lon, lat


def normalize_map(map, map_normalization):
    normalized = np.zeros(map.shape)
    for i in range(len(map)):
        if map_normalization[i] != 0:
            normalized[i] = map[i] / map_normalization[i]
    return normalized


def get_kids_parts(objects):
    objects_in_parts = [
        objects.loc[(objects['GAL_LAT'] > 0) & ((objects['GAL_LONG'] > 300) | (objects['GAL_LONG'] < 100))],
        objects.loc[(objects['GAL_LAT'] > 0) & (objects['GAL_LONG'] > 250) & (objects['GAL_LONG'] < 300)],
        objects.loc[(objects['GAL_LAT'] > 0) & (objects['GAL_LONG'] > 100) & (objects['GAL_LONG'] < 250)],
        objects.loc[(objects['GAL_LAT'] < 0) & (objects['GAL_LONG'] < 100)],
        objects.loc[(objects['GAL_LAT'] < 0) & (objects['GAL_LONG'] > 100)],
    ]
    return objects_in_parts


def show_correlations(maps_x, maps_y):
    r_df, p_df, c_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for map_x_name, map_x in maps_x:
        for map_y_name, map_y in maps_y:
            i_non_zero = np.nonzero(map_x)
            r, p = scipy.stats.pearsonr(map_x[i_non_zero], map_y[i_non_zero])
            c = np.corrcoef(map_x[i_non_zero], map_y[i_non_zero])[0][1]

            r_df.loc[map_x_name, map_y_name] = r
            p_df.loc[map_x_name, map_y_name] = p
            c_df.loc[map_x_name, map_y_name] = c

    print('pearson r')
    display(r_df)
    print('pearson p')
    display(p_df)
    print('correlation coefficient')
    display(c_df)


def get_column_desc(column_data):
    values, counts = np.unique(column_data, return_counts=True)
    s = sum(counts)
    contribution = counts / s * 100
    desc = '\n'.join(['{} - {} ({:.2f}%)'.format(v, c, p) for v, c, p in zip(values, counts, contribution)])
    return desc


def print_column_desc(column_data):
    print(get_column_desc(column_data))


def pretty_print_magnitude(str):
    return str.split('_')[-1] + ' magnitude'


def pretty_print_mags_combination(str):
    m_1 = str.split('_')[-2]
    m_2 = str.split('_')[-1]
    combination = str.split('_')[0]
    combination_symbol = {'COLOUR': '-', 'RATIO': '/'}[combination]
    return '{}{}{} {}'.format(m_1, combination_symbol, m_2, combination.lower())


def pretty_print_feature(str):
    if str == 'CLASS_STAR':
        return 'stellarity index'
    elif str.startswith('MAG'):
        return pretty_print_magnitude(str)
    elif str.startswith('COLOUR') or str.startswith('RATIO'):
        return pretty_print_mags_combination(str)
    else:
        return str


def get_external_qso_short_name(full_name):
    short_names = {
        'x 2QZ/6QZ': 'x 2QZ',
        'x Richards 2009': 'x R09',
        'x Richards 2015': 'x R15',
        'x DiPompeo 2015': 'x DP15',
    }
    return short_names[full_name]


def save_predictions(predictions_df, exp_name, timestamp):
    predictions_path = 'outputs/exp_preds/{exp_name}__{timestamp}.csv'.format(exp_name=exp_name,
                                                                              timestamp=timestamp)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info('predictions saved to: {}'.format(predictions_path))


def save_model(model, exp_name, timestamp):
    # TODO: neural net saving
    model_path = 'outputs/exp_models/{exp_name}__{timestamp}.joblib'.format(exp_name=exp_name,
                                                                            timestamp=timestamp)
    joblib.dump(model, model_path)
    logger.info('model saved to: {}'.format(model_path))


def save_catalog(catalog, exp_name, timestamp):
    logger.info('Saving catalog..')
    catalog_path = path.join(DATA_PATH, 'KiDS/DR4/catalogs/{exp_name}__{timestamp}.csv'.format(exp_name=exp_name,
                                                                                               timestamp=timestamp))
    catalog.to_csv(catalog_path, index=False)
    logger.info('catalog saved to: {}'.format(catalog_path))
