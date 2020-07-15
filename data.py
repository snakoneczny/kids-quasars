import os
import random
from collections import OrderedDict

import pandas as pd
from astropy.table import Table

from env_config import DATA_PATH

BASE_CLASSES = ['QSO', 'STAR', 'GALAXY']

COLUMNS_KIDS = ['ID', 'RAJ2000', 'DECJ2000', 'Flag', 'IMAFLAGS_ISO', 'MASK', 'CLASS_STAR', 'SG2DPHOT', 'Z_B']
COLUMNS_SDSS = ['CLASS', 'SUBCLASS', 'Z', 'Z_ERR', 'ZWARNING']
COLUMNS_GAIA = ['parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']

BITMAP_LENGTHS = {
    'Flag': 8,
    'IMAFLAGS_ISO': 7,
    'SG2DPHOT': 3,
    'MASK': 16,
}


def get_mag_str(band):
    return 'MAG_GAAP_{}'.format(band)


def get_color_str(band_1, band_2):
    return 'COLOUR_GAAP_{}_{}'.format(band_1, band_2)


def get_ratio_str(band_1, band_2):
    return 'RATIO_GAAP_{}_{}'.format(band_1, band_2)


def get_magerr_str(band):
    return 'MAGERR_GAAP_{}'.format(band)


def get_flag_gaap_str(band):
    return 'FLAG_GAAP_{}'.format(band)


BANDS = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']
BAND_NEXT_PAIRS = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'Z'), ('Z', 'Y'), ('Y', 'J'), ('J', 'H'), ('H', 'Ks')]


def get_mag_gaap_cols(bands=BANDS):
    return [get_mag_str(band) for band in bands]


def get_color_cols(band_tuples=BAND_NEXT_PAIRS):
    return [get_color_str(band_1, band_2) for band_1, band_2 in band_tuples]


def get_ratio_cols(band_tuples=BAND_NEXT_PAIRS):
    return [get_ratio_str(band_1, band_2) for band_1, band_2 in band_tuples]


def get_magerr_gaap_cols(bands=BANDS):
    return [get_magerr_str(band) for band in bands]


def get_flags_gaap_cols(bands=BANDS):
    return [get_flag_gaap_str(band) for band in bands]


def get_next_pairs(bands):
    return [(bands[i], bands[i + 1]) for i in range(len(bands) - 1)]


def get_all_pairs(bands):
    pairs = []
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            pairs.append((bands[i], bands[j]))
    return pairs


BAND_COLUMNS = get_mag_gaap_cols(BANDS)
BAND_ERR_COLUMNS = get_magerr_gaap_cols(BANDS)
COLOR_COLUMNS = get_color_cols(get_all_pairs(BANDS))
COLOR_NEXT_COLUMNS = get_color_cols(get_next_pairs(BANDS))
RATIO_COLUMNS = get_ratio_cols(get_all_pairs(BANDS))
FLAGS_GAAP_COLUMNS = get_flags_gaap_cols(BANDS)

# Colors and ratios with feature importance higher than 1% for classification task
PAIRS_CLF = [('J', 'Ks'), ('Y', 'Ks'), ('Z', 'Ks'), ('H', 'Ks'), ('Z', 'H'), ('Y', 'H'), ('u', 'r'), ('i', 'Ks'),
             ('J', 'H'), ('u', 'i'), ('Z', 'J'), ('u', 'g'), ('r', 'Ks'), ('r', 'J'), ('r', 'H'), ('g', 'i'),
             ('u', 'Y'), ('g', 'r'), ('i', 'H'), ('g', 'Y'), ('i', 'J'), ('u', 'Z'), ('u', 'H'), ('g', 'J'), ]
COLOR_COLUMNS_CLF = [get_color_str(x, y) for x, y in PAIRS_CLF]
RATIO_COLUMNS_CLF = [get_ratio_str(x, y) for x, y in PAIRS_CLF]

# Colors and ratios with feature importance higher than 1% for redshift estimation
PAIRS_REG = [('u', 'r'), ('Z', 'Ks'), ('g', 'i'), ('u', 'g'), ('g', 'Y'), ('g', 'Z'), ('H', 'Ks'), ('Y', 'J'),
             ('i', 'Ks'), ('r', 'i'), ('r', 'Y'), ('Z', 'J'), ('Z', 'Y'), ('g', 'r'), ('g', 'H'), ('Z', 'J')]
COLOR_COLUMNS_REG = [get_color_str(x, y) for x, y in PAIRS_REG]
RATIO_COLUMNS_REG = [get_ratio_str(x, y) for x, y in PAIRS_REG]

COLUMNS_KIDS_ALL = COLUMNS_KIDS + BAND_COLUMNS + COLOR_NEXT_COLUMNS + BAND_ERR_COLUMNS + FLAGS_GAAP_COLUMNS


def get_band_features(bands):
    features = get_mag_gaap_cols(bands)
    band_tuples = [(bands[i], bands[i + 1]) for i in range(len(bands) - 1)]
    features += get_color_cols(band_tuples)
    features += get_ratio_cols(band_tuples)
    return features


FEATURES = {
    'all': BAND_COLUMNS + COLOR_COLUMNS + RATIO_COLUMNS + ['CLASS_STAR', 'SG2DPHOT_3'],
    'no-sg': BAND_COLUMNS + COLOR_COLUMNS + RATIO_COLUMNS,
    'colors': COLOR_COLUMNS + RATIO_COLUMNS,
    'top-clf': BAND_COLUMNS + COLOR_COLUMNS_CLF + RATIO_COLUMNS_CLF + ['CLASS_STAR', 'SG2DPHOT_3'],
    'top-reg': BAND_COLUMNS + COLOR_COLUMNS_REG + RATIO_COLUMNS_REG + ['CLASS_STAR', 'SG2DPHOT_3'],
    'no-u': get_band_features(['g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']) + ['CLASS_STAR', 'SG2DPHOT_3'],
}

EXTERNAL_QSO = [
    (
        'x 2QZ/6QZ',
        os.path.join(DATA_PATH, 'KiDS/DR4/KiDS.DR4.x.6QZ.fits'),
        [get_mag_str('r'), 'ID', 'id1'],
    ),
    (
        'x Richards 2009',
        os.path.join(DATA_PATH, 'KiDS/DR4/KiDS.DR4.x.QSO.Richards.2009.fits'),
        [get_mag_str('r'), 'ID_1']
    ),
    (
        'x Richards 2015',
        os.path.join(DATA_PATH, 'KiDS/DR4/KiDS.DR4.x.QSO.Richards.2015.fits'),
        [get_mag_str('r'), 'ID']
    ),
    (
        'x DiPompeo 2015',
        os.path.join(DATA_PATH, 'KiDS/DR4/KiDS.DR4.x.QSO.GALEX.fits'),
        [get_mag_str('r'), 'ID_1', 'PQSO']
    ),
]


def process_kids(path, columns=None, bands=BANDS, kids_cleaning=True, sdss_cleaning=False, cut=None, n=None,
                 with_print=True, update_kids=False):
    extension = path.split('.')[-1]
    if extension == 'fits':
        data = read_fits_to_pandas(path, columns=columns, n=n)
    elif extension == 'csv':
        skiprows = get_skiprows(path, n) if n is not None else None
        data = pd.read_csv(path, usecols=columns, skiprows=skiprows)
    else:
        raise (Exception('Not supported file type {} in {}'.format(extension, path)))

    # Update with KiDS DR4.1
    if update_kids:
        data_update = read_fits_to_pandas(os.path.join(DATA_PATH, 'KiDS/DR4/KiDS.DR4.1.update.0.fits'),
                                          columns=[col for col in columns if col in COLUMNS_KIDS_ALL])
        data = data.set_index('ID', drop=False)
        data_update = data_update.set_index('ID', drop=False)
        data.update(data_update)

    return process_kids_data(data, bands=bands, kids_cleaning=kids_cleaning, sdss_cleaning=sdss_cleaning, cut=cut,
                             with_print=with_print)


def read_fits_to_pandas(filepath, columns=None, n=None):
    table = Table.read(filepath, format='fits')

    # Get first n rows if limit specified
    if n:
        table = table[0:n]

    # Get proper columns into a pandas data frame
    if columns:
        table = table[columns]
    table = table.to_pandas()

    # Astropy table assumes strings are byte arrays
    for col in ['ID', 'ID_1', 'CLASS', 'CLASS_PHOTO', 'id1']:
        if col in table and hasattr(table.loc[0, col], 'decode'):
            table.loc[:, col] = table[col].apply(lambda x: x.decode('UTF-8').strip())

    # Change type to work with it as with a bit map
    if 'IMAFLAGS_ISO' in table:
        table.loc[:, 'IMAFLAGS_ISO'] = table['IMAFLAGS_ISO'].astype(int)  # TODO: why? was okay earlier

    return table


def process_kids_data(data, bands=BANDS, cut=None, kids_cleaning=True, sdss_cleaning=False, with_print=True):
    if with_print:
        print('Data shape: {}'.format(data.shape))

    if kids_cleaning:
        data = clean_kids(data, bands=bands, with_print=with_print)

    if sdss_cleaning:
        data = clean_sdss(data)

    if cut:
        data = CUT_FUNCTIONS[cut](data, with_print=with_print)

    data = add_colors(data)
    data = add_magnitude_ratio(data)
    data = process_bitmaps(data)

    return data.reset_index(drop=True)


def get_skiprows(path, n):
    random.seed(8538)
    n_rows = sum(1 for _ in open(path)) - 1  # number of records in file (excludes header)
    skiprows = sorted(random.sample(range(1, n_rows + 1), n_rows - n))  # the 0-indexed header will not be included
    return skiprows


def clean_kids(data, bands=BANDS, with_print=True):
    # Drop NANs
    band_columns = get_mag_gaap_cols(bands)
    data_no_na = data.dropna(subset=band_columns).reset_index(drop=True)

    if with_print:
        n_left = data_no_na.shape[0]
        p_left = data_no_na.shape[0] / data.shape[0] * 100
        print('Droping NANs: {} ({:.2f}%) left'.format(n_left, p_left))

    mask = [True] * data_no_na.shape[0]

    # Remove errors
    # magerr_gaap_columns = get_magerr_gaap_cols(bands)
    # for c in magerr_gaap_columns:
    #     mask &= (data_no_na[c] < 1)
    # if with_print:
    #     n_left = mask.sum()
    #     p_left = mask.sum() / data.shape[0] * 100
    #     print('Removing errors bigger than 1: {} ({:.2f}%) left'.format(n_left, p_left))

    # Survey limiting magnitudes
    # mask &= (
    #         (data_no_na['MAG_GAAP_U'] < 24.3) &
    #         (data_no_na['MAG_GAAP_G'] < 25.1) &
    #         (data_no_na['MAG_GAAP_R'] < 24.9) &
    #         (data_no_na['MAG_GAAP_I'] < 23.8)
    # )
    # if with_print:
    #     n_left = mask.sum()
    #     p_left = mask.sum() / data.shape[0] * 100
    #     print('Removing limiting magnitudes: {} ({:.2f}%) left'.format(n_left, p_left))

    # Remove flags
    # flags_gaap = get_flags_gaap_cols(bands)
    # for c in flags_gaap:
    #     mask &= (data_no_na[c] == 0)
    # if with_print:
    #     n_left = mask.sum()
    #     p_left = mask.sum() / data.shape[0] * 100
    #     print('Removing GAAP flags: {} ({:.2f}%) left'.format(n_left, p_left))

    # Remove ima-flags
    # flag_mask = 0b0111111
    # mask &= (data_no_na['IMAFLAGS_ISO'] & flag_mask == 0)
    # if with_print:
    #     n_left = mask.sum()
    #     p_left = mask.sum() / data.shape[0] * 100
    #     print('Removing IMA flags: {} ({:.2f}%) left'.format(n_left, p_left))

    return data_no_na.loc[mask].reset_index(drop=True)


def cut_r(data, with_print=True):
    data_cut = data.loc[data[get_mag_str('r')] < 22].reset_index(drop=True)
    if with_print: print('Removing R > 22: {} left'.format(data_cut.shape[0]))
    return data_cut


def cut_u_g(data, with_print=True):
    data_cut = data.loc[data[get_color_str('u', 'g')] > 0].reset_index(drop=True)
    if with_print: print('Removing U-G < 0: {} left'.format(data_cut.shape[0]))
    return data_cut


CUT_FUNCTIONS = {
    'r': cut_r,
    'u-g': cut_u_g,
}


def add_colors(data):
    band_pairs = get_all_pairs(BANDS)
    for band_x, band_y in band_pairs:
        column_x = get_mag_str(band_x)
        column_y = get_mag_str(band_y)
        color_str = get_color_str(band_x, band_y)
        if color_str not in data.columns:
            data.loc[:, color_str] = data[column_x] - data[column_y]
    return data


def add_magnitude_ratio(data):
    band_pairs = get_all_pairs(BANDS)
    for band_x, band_y in band_pairs:
        column_x = get_mag_str(band_x)
        column_y = get_mag_str(band_y)
        data.loc[:, get_ratio_str(band_x, band_y)] = data[column_x] / data[column_y]
    return data


def process_bitmaps(data, bitmap_cols=None):
    bitmap_cols = BITMAP_LENGTHS.keys() if not bitmap_cols else bitmap_cols
    for bitmap_col in bitmap_cols:
        if bitmap_col not in data:
            continue
        data.loc[:, bitmap_col] = data[bitmap_col].astype(int)
        for i in range(BITMAP_LENGTHS[bitmap_col]):
            position = 2 ** i
            data.loc[:, '{}_{}'.format(bitmap_col, i + 1)] = ((data[bitmap_col] & position) != 0).astype(int)
    return data


def clean_sdss(data, with_print=True):
    data_cleaned = data.loc[data['ZWARNING'] == 0].reset_index(drop=True)
    if with_print:
        print('Cleaning SDSS: {} left'.format(data_cleaned.shape[0]))
    return data_cleaned


def process_gaia(data, parallax_error=1, pm_error=None, parallax_lim=None, pm_lim=None, with_print=True):
    # Get 5 position observations (parallax should be enough)
    movement_mask = ~data[['parallax']].isnull().any(axis=1)
    data = data.loc[movement_mask]
    if with_print:
        print('5 position shape: {}'.format(data.shape))

    data = norm_gaia_observations(data)
    data = clean_gaia(data, parallax_error=parallax_error, pm_error=pm_error, parallax_lim=parallax_lim, pm_lim=pm_lim)
    return data


def clean_gaia(data, parallax_error=1, pm_error=None, parallax_lim=None, pm_lim=None):
    if parallax_error:
        data = data.loc[data['parallax_error'] < parallax_error]
        print('Removing paralax_error: {}'.format(data.shape))

    if pm_error:
        data = data.loc[data['pmra_error'] < pm_error]
        print('Removing pmra_error: {}'.format(data.shape))

        data = data.loc[data['pmdec_error'] < pm_error]
        print('Removing pmdec_error: {}'.format(data.shape))

    if parallax_lim:
        data = data.loc[(data['parallax'] > parallax_lim[0]) & (data['parallax'] < parallax_lim[1])]
        print('Removing limiting parallax: {}'.format(data.shape))

    if pm_lim:
        proper_motion_mask = (data['pmra'].pow(2) + data['pmdec'].pow(2) < pm_lim)
        data = data.loc[proper_motion_mask]
        print('Removing limiting pmra and pmdec: {}'.format(data.shape))

    return data


def norm_gaia_observations(data):
    for col in ['parallax', 'pmra', 'pmdec']:
        data.loc[:, col + '_norm'] = data[col] / data[col + '_error']
    return data


def process_2df(data):
    data.loc[:, 'id1'] = data['id1'].apply(lambda x: x.strip().upper())
    data = data.replace('GAL', 'GALAXY')
    return data


def merge_specialized_catalogs(ctlg_clf, ctlg_z_qso, ctlg_z_galaxy=None):
    if isinstance(ctlg_clf, str):
        ctlg_clf = read_fits_to_pandas(ctlg_clf)
    if isinstance(ctlg_z_qso, str):
        ctlg_z_qso = read_fits_to_pandas(ctlg_z_qso)
    if isinstance(ctlg_z_galaxy, str):
        ctlg_z_galaxy = read_fits_to_pandas(ctlg_z_galaxy)

    catalog = ctlg_clf.copy()

    # QSO
    catalog.loc[catalog['CLASS_PHOTO'] == 'QSO', 'Z_PHOTO'] = ctlg_z_qso.loc[catalog['CLASS_PHOTO'] == 'QSO', 'Z_PHOTO']
    if 'Z_PHOTO_STDDEV' in ctlg_z_qso:
        catalog.loc[catalog['CLASS_PHOTO'] == 'QSO', 'Z_PHOTO_STDDEV'] = ctlg_z_qso.loc[
            catalog['CLASS_PHOTO'] == 'QSO', 'Z_PHOTO_STDDEV']

    # Galaxy
    if ctlg_z_galaxy:
        catalog.loc[catalog['CLASS_PHOTO'] == 'GALAXY', 'Z_PHOTO'] = ctlg_z_galaxy.loc[
            catalog['CLASS_PHOTO'] == 'GALAXY', 'Z_PHOTO']
        if 'Z_PHOTO_STDDEV' in ctlg_z_galaxy:
            catalog.loc[catalog['CLASS_PHOTO'] == 'GALAXY', 'Z_PHOTO_STDDEV'] = ctlg_z_galaxy.loc[
                catalog['CLASS_PHOTO'] == 'GALAXY', 'Z_PHOTO_STDDEV']

    # Star
    catalog.loc[catalog['CLASS_PHOTO'] == 'STAR', 'Z_PHOTO'] = 0
    catalog.loc[catalog['CLASS_PHOTO'] == 'STAR', 'Z_PHOTO_STDDEV'] = 0

    return catalog


def add_subset_info(data, extra_info=False):
    cs_safe_idx = (data['CLASS_STAR'] > 0.8) | (data['CLASS_STAR'] < 0.2)
    if extra_info:
        subsets_idx = [
            ('extrap., r in (24, 25)', cs_safe_idx & (data['MAG_GAAP_r'] < 25)),
            ('extrap., r in (23, 24)', cs_safe_idx & (data['MAG_GAAP_r'] < 24)),
            ('extrap., r in (22, 23)', cs_safe_idx & (data['MAG_GAAP_r'] < 23)),
            ('safe, r < 22', cs_safe_idx & (data['MAG_GAAP_r'] < 22)),
        ]
    else:
        subsets_idx = [
            ('extrapolation', cs_safe_idx & (data['MAG_GAAP_r'] < 25)),
            ('safe', cs_safe_idx & (data['MAG_GAAP_r'] < 22)),
        ]
    data['subset'] = 'unsafe'
    for subset_name, subset_idx in subsets_idx:
        data.loc[subset_idx, 'subset'] = subset_name
    return data


def get_joint_inference_subsets(data):
    data = add_subset_info(data)
    subsets = OrderedDict([
        ('KiDS', data),
        ('KiDS extrapolation', data.loc[data['subset'].isin(['safe', 'extrapolation'])]),
        ('KiDS safe', data.loc[data['subset'] == 'safe']),
    ])
    return subsets


def get_inference_subsets(data):
    data = add_subset_info(data)
    subsets = OrderedDict(
        [(subset_name, data.loc[data['subset'] == subset_name]) for subset_name in
         ['safe', 'extrapolation', 'unsafe']])
    return subsets
