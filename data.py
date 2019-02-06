import random
from collections import OrderedDict

import numpy as np
import pandas as pd

EXTERNAL_QSO_PATHS = [
    '/media/snakoneczny/data/KiDS/KiDS.DR3.x.2QZ6QZ.cols.csv',
    '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.RICHARDS.2009.csv',
    '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.RICHARDS.2015.csv',
    '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.GALEX.csv',
]

EXTERNAL_QSO_DICT = OrderedDict(
    zip(['x 2QZ/6QZ', 'x Richards 2009', 'x Richards 2015', 'x DiPompeo 2015'], EXTERNAL_QSO_PATHS))

BASE_CLASSES = ['QSO', 'STAR', 'GALAXY']

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

RATIO_U_G = 'RATIO_U_G'
RATIO_U_R = 'RATIO_U_R'
RATIO_U_I = 'RATIO_U_I'
RATIO_G_R = 'RATIO_G_R'
RATIO_G_I = 'RATIO_G_I'
RATIO_R_I = 'RATIO_R_I'

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

RATIO_COLUMNS = [
    RATIO_U_G,
    RATIO_U_R,
    RATIO_U_I,
    RATIO_G_R,
    RATIO_G_I,
    RATIO_R_I,
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

FLAGS = ['FLAG_U', 'FLAG_G', 'FLAG_R', 'FLAG_I']
IMA_FLAGS = ['IMAFLAGS_ISO_U', 'IMAFLAGS_ISO_G', 'IMAFLAGS_ISO_R', 'IMAFLAGS_ISO_I']

FEATURES = {
    'all': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS, RATIO_COLUMNS, ['CLASS_STAR']]),
    'no-mags': np.concatenate([COLOR_COLUMNS, RATIO_COLUMNS, ['CLASS_STAR']]),
    'magnitudes-colors-cstar': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS, ['CLASS_STAR']]),
    'magnitudes-colors': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS]),
    'colors': COLOR_COLUMNS,
    'colors-cstar': np.concatenate([COLOR_COLUMNS, ['CLASS_STAR']]),
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

    if sdss_cleaning:
        data = clean_sdss(data)

    data = calib_mag_for_ext(data)

    if cut:
        data = CUT_FUNCTIONS[cut](data, with_print=with_print)

    data = add_magnitude_ratio(data)

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
    if with_print:
        n_left = data_no_na.shape[0]
        p_left = data_no_na.shape[0] / data.shape[0] * 100
        print('Droping NANs: {} ({:.2f}%) left'.format(n_left, p_left))

    mask = [1] * data_no_na.shape[0]

    # Remove errors
    for b in BANDS:
        mask &= (data_no_na['MAGERR_GAAP_{}'.format(b)] < 1)
    if with_print:
        n_left = mask.sum()
        p_left = mask.sum() / data.shape[0] * 100
        print('Removing errors bigger than 1: {} ({:.2f}%) left'.format(n_left, p_left))

    # Survey limiting magnitudes
    mask &= (
            (data_no_na['MAG_GAAP_U'] < 24.3) &
            (data_no_na['MAG_GAAP_G'] < 25.1) &
            (data_no_na['MAG_GAAP_R'] < 24.9) &
            (data_no_na['MAG_GAAP_I'] < 23.8)
    )
    if with_print:
        n_left = mask.sum()
        p_left = mask.sum() / data.shape[0] * 100
        print('Removing limiting magnitudes: {} ({:.2f}%) left'.format(n_left, p_left))

    # Remove flags
    # for c in FLAGS:
    #     mask &= (data_no_na[c] == 0)
    # if with_print: print('Removing flags: {} left'.format(mask.sum()))

    # Remove ima-flags
    flag_mask = 0b01111111
    for c in IMA_FLAGS:
        mask &= (data_no_na[c] & flag_mask == 0)
    if with_print:
        n_left = mask.sum()
        p_left = mask.sum() / data.shape[0] * 100
        print('Removing IMA flags: {} ({:.2f}%) left'.format(n_left, p_left))

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


def add_magnitude_ratio(data):
    for column_x, column_y in BAND_PAIRS:
        band_x = column_x.split('_')[-1]
        band_y = column_y.split('_')[-1]
        data['RATIO_{}_{}'.format(band_x, band_y)] = data[column_x] / data[column_y]
    return data


def clean_sdss(data, with_print=True):
    data_cleaned = data.loc[data['ZWARNING'] == 0].reset_index(drop=True)
    if with_print: print('Cleaning SDSS: {} left'.format(data_cleaned.shape[0]))
    return data_cleaned


def process_gaia(data, parallax_error=1, pm_error=None, parallax_lim=None, pm_lim=None, with_print=True):
    # Get 5 position observations (parallax should suffice)
    movement_mask = ~data[['parallax']].isnull().any(axis=1)
    data = data.loc[movement_mask]
    if with_print: print('5 position shape: {}'.format(data.shape))

    data = norm_gaia_observations(data)
    data = clean_gaia(data, parallax_error=parallax_error, pm_error=pm_error, parallax_lim=parallax_lim, pm_lim=pm_lim,
                      with_print=with_print)
    return data


def clean_gaia(data, parallax_error=1, pm_error=None, parallax_lim=None, pm_lim=None, with_print=True):
    if parallax_error:
        data = data.loc[data['parallax_error'] < parallax_error]
        if with_print: print('Removing paralax_error shape: {}'.format(data.shape))

    if pm_error:
        data = data.loc[data['pmra_error'] < pm_error]
        if with_print: print('Removing pmra_error shape: {}'.format(data.shape))

        data = data.loc[data['pmdec_error'] < pm_error]
        if with_print: print('Removing pmdec_error shape: {}'.format(data.shape))

    if parallax_lim:
        data = data.loc[(data['parallax'] > parallax_lim[0]) & (data['parallax'] < parallax_lim[1])]
        if with_print: print('Removing parallax_norm shape: {}'.format(data.shape))

    if pm_lim:
        proper_motion_mask = (data['pmra'].pow(2) + data['pmdec'].pow(2) < pm_lim)
        data = data.loc[proper_motion_mask]
        if with_print: print('Removing pmra_norm and pmdec_norm shape: {}'.format(data.shape))

    return data


def norm_gaia_observations(data):
    for col in ['parallax', 'pmra', 'pmdec']:
        data[col + '_norm'] = data[col] / data[col + '_error']
    return data


def process_2df(data):
    data['id1'] = data['id1'].apply(lambda x: x.strip())
    data['id1'] = data['id1'].apply(lambda x: x.upper())
    data = data.replace('GAL', 'GALAXY')
    return data
