import random
import logging
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

BANDS = ['U', 'G', 'R', 'I']

MAG_GAAP_U = 'MAG_GAAP_U'
MAG_GAAP_G = 'MAG_GAAP_G'
MAG_GAAP_R = 'MAG_GAAP_R'
MAG_GAAP_I = 'MAG_GAAP_I'

MAG_GAAP_CALIB_U = 'MAG_GAAP_CALIB_U'
MAG_GAAP_CALIB_G = 'MAG_GAAP_CALIB_G'
MAG_GAAP_CALIB_R = 'MAG_GAAP_CALIB_R'
MAG_GAAP_CALIB_I = 'MAG_GAAP_CALIB_I'

COLOR_GAAPHOM_U_G = 'COLOR_GAAPHOM_U_G'
COLOR_GAAPHOM_U_R = 'COLOR_GAAPHOM_U_R'
COLOR_GAAPHOM_U_I = 'COLOR_GAAPHOM_U_I'
COLOR_GAAPHOM_G_R = 'COLOR_GAAPHOM_G_R'
COLOR_GAAPHOM_G_I = 'COLOR_GAAPHOM_G_I'
COLOR_GAAPHOM_R_I = 'COLOR_GAAPHOM_R_I'

BAND_COLUMNS = [
    MAG_GAAP_U,
    MAG_GAAP_G,
    MAG_GAAP_R,
    MAG_GAAP_I,
]

BAND_CALIB_COLUMNS = [
    MAG_GAAP_CALIB_U,
    MAG_GAAP_CALIB_G,
    MAG_GAAP_CALIB_R,
    MAG_GAAP_CALIB_I,
]

COLOR_COLUMNS = [
    COLOR_GAAPHOM_U_G,
    COLOR_GAAPHOM_U_R,
    COLOR_GAAPHOM_U_I,
    COLOR_GAAPHOM_G_R,
    COLOR_GAAPHOM_G_I,
    COLOR_GAAPHOM_R_I,
]

BAND_PAIRS = [
    (MAG_GAAP_CALIB_U, MAG_GAAP_CALIB_G),
    (MAG_GAAP_CALIB_U, MAG_GAAP_CALIB_R),
    (MAG_GAAP_CALIB_U, MAG_GAAP_CALIB_I),
    (MAG_GAAP_CALIB_G, MAG_GAAP_CALIB_R),
    (MAG_GAAP_CALIB_G, MAG_GAAP_CALIB_I),
    (MAG_GAAP_CALIB_R, MAG_GAAP_CALIB_I),
]

COLOR_PAIRS = [
    (COLOR_GAAPHOM_U_G, COLOR_GAAPHOM_G_R),
    (COLOR_GAAPHOM_G_R, COLOR_GAAPHOM_R_I),
    (COLOR_GAAPHOM_U_G, COLOR_GAAPHOM_R_I),
]

SE_FLAGS = ['FLAG_U', 'FLAG_G', 'FLAG_R', 'FLAG_I']
IMA_FLAGS = ['IMAFLAGS_ISO_U', 'IMAFLAGS_ISO_G', 'IMAFLAGS_ISO_R', 'IMAFLAGS_ISO_I']

FEATURES = {
    'colors': COLOR_COLUMNS,
    'all': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS])
}


def process_kids(path, sdss_cleaning=False, cut=None, n=None, with_print=True):
    if n is not None:
        data = read_random_sample(path, n)
    else:
        data = pd.read_csv(path)

    return process_kids_data(data, sdss_cleaning=sdss_cleaning, cut=cut, with_print=with_print)


def process_kids_data(data, sdss_cleaning=False, cut=None, with_print=True):
    if with_print: print('Data shape: {}'.format(data.shape))
    
    data = clean_kids(data, with_print)
    data = calib_mag_for_ext(data)
    
    if sdss_cleaning:
        data = clean_sdss(data)
    
    if cut:
        data = CUT_FUNCTIONS[cut](data, with_print=with_print)
    
    return data.reset_index(drop=True)


def read_random_sample(path, n):
    random.seed(8538)
    n_rows = sum(1 for _ in open(path)) - 1  # number of records in file (excludes header)
    skip = sorted(random.sample(range(1, n_rows + 1), n_rows - n))  # the 0-indexed header will not be included
    return pd.read_csv(path, skiprows=skip)


def add_sdss_info(data, sdss_path):
    data_sdss = pd.read_csv(sdss_path, usecols=['ID', 'CLASS', 'SUBCLASS', 'Z'])
    print('SDSS data shape: {}'.format(data_sdss.shape))

    data = data.merge(data_sdss, how='left', on='ID')
    data['CLASS_FILLED'] = data['CLASS'].fillna(value='UNKNOWN')

    data_sdss = data.dropna(axis=0, subset=['CLASS'])

    print('SDSS info shape: {}'.format(data_sdss.shape))
    print(np.unique(data['CLASS_FILLED'], return_counts=True))

    return data, data_sdss


def clean_kids(data, with_print=True):
    # Drop NANs
    data_no_na = data.dropna(subset=[MAG_GAAP_U, MAG_GAAP_G, MAG_GAAP_R, MAG_GAAP_I]).reset_index(drop=True)
    if with_print: print('Droping NANs: {} left'.format(data_no_na.shape[0]))

    mask = [1] * data_no_na.shape[0]

    # Survey limiting magnitudes
    mask &= (
            (data_no_na['MAG_GAAP_U'] < 24.3) &
            (data_no_na['MAG_GAAP_G'] < 25.1) &
            (data_no_na['MAG_GAAP_R'] < 24.9) &
            (data_no_na['MAG_GAAP_I'] < 23.8)
    )
    if with_print: print('Removing limiting magnitudes: {} left'.format(mask.sum()))

    # Remove errors
    for b in BANDS:
        mask &= (data_no_na['MAGERR_GAAP_{}'.format(b)] < 1)
    if with_print: print('Removing errors bigger than 1: {} left'.format(mask.sum()))

    # Remove flags
    for c in SE_FLAGS:
        mask &= (data_no_na[c] == 0)
    if with_print: print('Removing SExtractor flags: {} left'.format(mask.sum()))

    # Remove ima-flags
    flag_mask = 0b01111111
    for c in IMA_FLAGS:
        mask &= (data_no_na[c] & flag_mask == 0)
    if with_print: print('Removing ima-flags: {} left'.format(mask.sum()))

    # Tile flag
    # mask &= (data_no_na['TILE_FLAG'] == 0)
    # if with_print: print('Removing tile flag: {} left'.format(mask.sum()))

    return data_no_na.loc[mask].reset_index(drop=True)


def calib_mag_for_ext(data):
    for b in BANDS:
        data['MAG_GAAP_CALIB_{}'.format(b)] = data['MAG_GAAP_{}'.format(b)] + \
                                              data['ZPT_OFFSET_{}'.format(b)] - \
                                              data['EXT_SFD_{}'.format(b)]
    return data


def clean_sdss(data, with_print=True):
    data_cleaned = data.loc[data['ZWARNING'] == 0].reset_index(drop=True)
    if with_print: print('Cleaning SDSS: {} left'.format(data_cleaned.shape[0]))
    return data_cleaned


def cut_r(data, with_print=True):
    data_cut = data.loc[data[MAG_GAAP_CALIB_R] < 22].reset_index(drop=True)
    if with_print: print('Removing R > 22: {} left'.format(data_cut.shape[0]))
    return data_cut


def cut_u_g(data, with_print=True):
    data_cut = data.loc[data[COLOR_GAAPHOM_U_G] > 0].reset_index(drop=True)
    if with_print: print('Removing U-G < 0: {} left'.format(data_cut.shape[0]))
    return data_cut


CUT_FUNCTIONS = {
    'r': cut_r,
    'u-g': cut_u_g,
}


def describe_column(data):
    values, counts = np.unique(data, return_counts=True)
    s = sum(counts)
    contribution = counts / s * 100
    return values, counts, contribution


def print_rf_feature_ranking(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    logger.info('feature ranking')
    for f in range(X.shape[1]):
        print('%d. feature %s (%f)' % (f + 1, X.columns[indices[f]], importances[indices[f]]))


def number_count_analysis(ds, c=10):
    for b in BAND_CALIB_COLUMNS:
        
        m_min = math.ceil(ds[b].min()) + 1
        m_max = math.ceil(ds[b].max()) - 1
        
        x, y, y_norm = [], [], []
        for m in range(m_min, m_max + 1):
            x.append(m)
            v = ds.loc[ds[b] < m].shape[0]
            if v != 0:
                v = math.log(v, 10)
            y.append(v)
            
            y_norm.append(0.6 * m - c)

        plt.figure()
        plt.title(b)
        plt.xlabel('m')
        plt.plot(x, y, label='log N(≤ m)')
        plt.plot(x, y_norm, label='0.6 * m - {}'.format(c))
        plt.legend()
