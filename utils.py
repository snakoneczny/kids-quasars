import os
import random
import logging
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

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

SE_FLAGS = ['FLAG_U', 'FLAG_G', 'FLAG_R', 'FLAG_I']
IMA_FLAGS = ['IMAFLAGS_ISO_U', 'IMAFLAGS_ISO_G', 'IMAFLAGS_ISO_R', 'IMAFLAGS_ISO_I']

FEATURES = {
    'all': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS, RATIO_COLUMNS, ['CLASS_STAR']]),
    'magnitudes-colors': np.concatenate([BAND_CALIB_COLUMNS, COLOR_COLUMNS]),
    'colors': COLOR_COLUMNS,
    'colors-cstar': np.concatenate([COLOR_COLUMNS, ['CLASS_STAR']]),
}


def process_kids(path, subset=None, sdss_cleaning=False, cut=None, n=None, with_print=True):
    if n is not None:
        data = read_random_sample(path, n)
    else:
        data = pd.read_csv(path)

    return process_kids_data(data, subset=subset, sdss_cleaning=sdss_cleaning, cut=cut, with_print=with_print)


def process_kids_data(data, subset=None, sdss_cleaning=False, cut=None, with_print=True):
    if with_print: print('Data shape: {}'.format(data.shape))

    data = clean_kids(data, with_print)

    if subset:
        data = get_subset(data, subset, with_print)

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


def get_subset(data, subset, with_print=True):
    if subset == 'star':
        data_subset = data.loc[data['CLASS_STAR'] >= 0.5]
    elif subset == 'non-star':
        data_subset = data.loc[data['CLASS_STAR'] <= 0.5]
    else:
        raise (ValueError('{} is not an available subset'.format(subset)))

    if with_print: print('Extracting {} subset: {}'.format(subset, data_subset.shape[0]))

    return data_subset


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
    if with_print: print('Removing KiDS flags: {} left'.format(mask.sum()))

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


def add_magnitude_ratio(data):
    for column_x, column_y in BAND_PAIRS:
        band_x = column_x.split('_')[-1]
        band_y = column_y.split('_')[-1]
        data['RATIO_{}_{}'.format(band_x, band_y)] = data[column_x] / data[column_y]
    return data


def describe_column(data):
    values, counts = np.unique(data, return_counts=True)
    s = sum(counts)
    contribution = counts / s * 100
    return values, counts, contribution


def print_feature_ranking(model, X):
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


def qso_catalogs_report(catalog, save=False):
    qso_catalog_paths = [
        '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.RICHARDS.2009.csv',
        '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.RICHARDS.2015.csv',
        '/media/snakoneczny/data/KiDS/KiDS.DR3.x.QSO.GALEX.csv',
    ]

    print('Catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog['CLASS']))

    for qso_catalog_path in qso_catalog_paths:
        qso_catalog_report(qso_catalog_path, catalog, save=save)


def qso_catalog_report(qso_catalog_path, catalog, save=False):
    qso_catalog = pd.read_csv(qso_catalog_path, usecols=['ID'])
    is_in_qso = catalog['ID'].isin(qso_catalog['ID'])
    n_train_in_qso = sum(catalog.loc[is_in_qso, 'train'])

    print('--------------------')
    print(os.path.basename(qso_catalog_path))
    print('QSO catalog x KiDS size: {}'.format(qso_catalog.shape[0]))
    print('QSO catalog x KiDS catalog size: {}, train elements: {}'.format(sum(is_in_qso), n_train_in_qso))
    print(describe_column(catalog.loc[is_in_qso, 'CLASS']))

    if save:
        catalog.loc[is_in_qso].to_csv('catalogs_intersection/{}.csv'.format(os.path.basename(qso_catalog_path)))


def r_train_test_split(*args, train_val, test):
    splitted_list = []
    for arg in args:
        splitted_list.extend([arg[train_val], arg[test]])
    return splitted_list


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
