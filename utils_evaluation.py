import math
import os
from collections.__init__ import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from utils import BAND_CALIB_COLUMNS, describe_column, EXTERNAL_QSO_PATHS, clean_gaia


def metric_class_split(y_true, y_pred, classes, metric):
    scores = []
    for c in np.unique(classes):  # TODO: merge with encoder
        print(c)
        c_idx = np.array((classes == c))
        scores.append(metric(y_true[c_idx], y_pred[c_idx]))
    return scores


def number_count_analysis(ds, c=10, linear_range=(18, 20)):
    for b in BAND_CALIB_COLUMNS:

        m_min = int(math.ceil(ds[b].min()) + 1)
        m_max = int(math.ceil(ds[b].max()) - 1)

        x, y = [], []
        for m in range(m_min, m_max + 1):
            x.append(m)
            v = ds.loc[ds[b] < m].shape[0]
            if v != 0:
                v = math.log(v, 10)
            y.append(v)

        plt.plot(x, y, label=b)

    x_linear = np.arange(linear_range[0], linear_range[1] + 0.1, 0.1)
    y_linear = [0.6 * m - c for m in x_linear]
    plt.plot(x_linear, y_linear, label='0.6 * m - {}'.format(c))
    plt.xlabel('m')
    plt.ylabel('log N(≤ m)')
    plt.legend()


def test_external_qso(catalog, save=False):
    print('catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog['CLASS']))

    for external_path in EXTERNAL_QSO_PATHS:
        external_catalog = pd.read_csv(external_path, usecols=['ID'])
        title = os.path.basename(external_path)[:-4]
        test_against_external_catalog(external_catalog, catalog, title=title, save=save)


def test_gaia(catalog, catalog_x_gaia_path, class_column='CLASS', id_column='ID', save=False):
    print('catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog[class_column]))

    catalog_x_gaia = pd.read_csv(catalog_x_gaia_path)

    movement_mask = ~catalog_x_gaia[['parallax', 'pmdec', 'pmra']].isnull().any(axis=1)
    catalog_x_gaia_movement = catalog_x_gaia.loc[movement_mask]

    catalog_x_gaia_movement = clean_gaia(catalog_x_gaia_movement)

    test_against_external_catalog(catalog_x_gaia_movement, catalog, class_column=class_column, id_column=id_column,
                                  title='GAIA', save=save)


def test_against_external_catalog(ext_catalog, catalog, class_column='CLASS', id_column='ID', title='', save=False):
    is_in_ext = catalog[id_column].isin(ext_catalog[id_column])
    catalogs_cross = catalog.loc[is_in_ext]
    n_train_in_ext = sum(catalogs_cross['train']) if 'train' in catalogs_cross.columns else 0

    print('--------------------')
    print(title)
    print('ext. catalog x base set size: {}'.format(ext_catalog.shape[0]))
    print('ext. catalog x base catalog size: {}, train elements: {}'.format(sum(is_in_ext), n_train_in_ext))
    print('catalogs cross:')
    print(describe_column(catalogs_cross[class_column]))

    if 'train' in catalogs_cross.columns:
        catalogs_cross_no_train = catalogs_cross.loc[catalogs_cross['train'] == 0]
        print('catalogs cross, no train:')
        print(describe_column(catalogs_cross_no_train[class_column]))

    # Plot class histograms
    if 'MAG_GAAP_CALIB_U' in catalogs_cross.columns:
        for c in BAND_CALIB_COLUMNS:
            plt.figure()
            for t in ['STAR', 'GALAXY', 'QSO']:
                sns.distplot(catalogs_cross.loc[catalogs_cross['CLASS'] == t][c], label=t, kde=False, rug=False,
                             hist_kws={'alpha': 0.5})
                plt.title(title)
            plt.legend()

    if save:
        catalog.loc[is_in_ext].to_csv('catalogs_intersection/{}.csv'.format(title))


def gaia_motion_analysis(data, norm=True):
    movement_mask = ~data[['parallax', 'pmdec', 'pmra']].isnull().any(axis=1)
    data_movement = data.loc[movement_mask]

    classes = ['QSO', 'GALAXY', 'STAR']
    motions = ['parallax', 'pmra', 'pmdec']
    if norm: motions = [m + '_norm' for m in motions]
    mu_df = pd.DataFrame(index=classes, columns=motions)
    sigma_df = pd.DataFrame(index=classes, columns=motions)
    median_df = pd.DataFrame(index=classes, columns=motions)

    for class_name in classes:
        for motion in motions:
            data_of_interest = data_movement.loc[data_movement['CLASS'] == class_name][motion]
            (mu, sigma) = stats.norm.fit(data_of_interest)
            median = np.median(data_of_interest)

            mu_df.loc[class_name, motion] = mu
            sigma_df.loc[class_name, motion] = sigma
            median_df.loc[class_name, motion] = median

            plt.figure()
            sns.distplot(data_of_interest)
            plt.ylabel(class_name)

    print('Mean:')
    display(mu_df)
    print('Sigma:')
    display(sigma_df)
    print('Median:')
    display(median_df)


def proba_motion_analysis(data_x_gaia, motions=['parallax'], x_lim=(0.3, 1), step=0.02):
    mu_dict, sigma_dict, median_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    # Get QSOs
    qso_x_gaia = data_x_gaia.loc[data_x_gaia['CLASS'] == 'QSO']

    # Limit QSOs to proba thresholds
    thresholds = np.arange(x_lim[0], x_lim[1] + step, step)
    for thr in thresholds:
        qso_x_gaia_limited = qso_x_gaia.loc[qso_x_gaia['QSO'] >= thr]

        for motion in motions:
            # Get stats
            (mu, sigma) = stats.norm.fit(qso_x_gaia_limited[motion])
            median = np.median(qso_x_gaia_limited[motion])

            # Store values
            mu_dict[motion].append(mu)
            sigma_dict[motion].append(sigma)
            median_dict[motion].append(median)

    # Plot statistics
    to_plot = [(mu_dict, 'mu'), (sigma_dict, 'sigma'), (median_dict, 'median')]

    for t in to_plot:
        plt.figure()

        for motion in motions:
            plt.plot(thresholds, t[0][motion], label=motion)
            plt.xlabel('probability threshold')
            plt.ylabel(t[1])

        plt.legend()
