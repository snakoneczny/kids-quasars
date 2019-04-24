import itertools
from collections import OrderedDict

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import pretty_print_feature, \
    get_external_qso_short_name
from data import EXTERNAL_QSO_DICT, BASE_CLASSES, BAND_COLUMNS, process_2df

CUSTOM_COLORS = {
    'QSO': (0.08605633600581403, 0.23824692404212, 0.30561236308077167),
    'STAR': (0.7587183008012618, 0.7922069335474338, 0.9543861221913403),
    'GALAXY': (0.32927729263408284, 0.4762845556584382, 0.1837155549758328),
    'not SDSS': (0.8146245329198283, 0.49548316572322215, 0.5752525936416857),
    'UNKNOWN': (0.6, 0.6, 0.6),
    False: (0.8299576787894204, 0.5632024035248271, 0.7762744444444445),
    True: (0.1700423212105796, 0.43679759647517286, 0.22372555555555548),
}

LINE_STYLES = ['-', '--', '-.', ':']


def get_line_style(i):
    return LINE_STYLES[i % len(LINE_STYLES)]


def get_cubehelix_palette(n):
    n_color = 2 if n == 1 else n
    palette = sns.color_palette('cubehelix', n_color)
    palette.reverse()
    if n == 1:
        return palette[1:]
    else:
        return palette


def plot_embedding(embedding, labels, label='class', is_continuous=False, alpha=0.5, color_palette='cubehelix',
                   with_custom_colors=True, labels_in_order=False, legend_loc='upper right'):
    if not is_continuous:
        labels_unique = np.unique(labels)
        n_colors = len(labels_unique)
        if isinstance(color_palette, str):
            color_palette = sns.color_palette(color_palette, n_colors)
            color_palette = {labels_unique[k]: v for k, v in enumerate(color_palette)}
        if with_custom_colors: color_palette.update(CUSTOM_COLORS)

        data = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], label: labels})
        hue_order = color_palette.keys() if labels_in_order else None
        sns.lmplot(x='x', y='y', hue=label, data=data, palette=color_palette, fit_reg=False,
                   scatter_kws={'alpha': alpha}, size=7, hue_order=hue_order, legend=False)
        plt.legend(loc=legend_loc)

    else:
        cmap = sns.cubehelix_palette(as_cmap=True)
        f, ax = plt.subplots(figsize=(9, 7))
        points = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=50, cmap=cmap, alpha=alpha)
        cb = f.colorbar(points)
        cb.set_label(label)


def plot_confusion_matrix(cm, classes, normalize=False, true_label='SDSS'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    cmap = sns.cubehelix_palette(light=.95, as_cmap=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        str = format(cm[i, j], fmt)
        if normalize: str += '%'
        plt.text(j, i, str, horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('{} - true label'.format(true_label))
    plt.xlabel('KiDS - predicted label')


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(precisions, recalls, average_precision, precision, recall):
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')

    plt.plot(precision, recall, marker='o', markersize=3, color='red')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


def plot_histograms(data_dict, columns=BAND_COLUMNS, x_lim_dict=None, title=None, pretty_print_function=None,
                    legend_loc='upper left', legend_size=None):
    color_palette = get_cubehelix_palette(len(data_dict))
    for column in columns:

        plt.figure()
        for i, (label, data) in enumerate(data_dict.items()):
            sns.distplot(data[column], label=label, kde=False, rug=False, norm_hist=True, color=color_palette[i],
                         hist_kws={'alpha': 1.0, 'histtype': 'step', 'linewidth': 1.5, 'linestyle': get_line_style(i)})

        if x_lim_dict and column in x_lim_dict:
            plt.xlim(x_lim_dict[column][0], x_lim_dict[column][1])

        if title: plt.title(title)
        if pretty_print_function: plt.xlabel(pretty_print_function(column))
        plt.ylabel('normalized counts per bin')
        prop = {'size': legend_size} if legend_size else {}
        plt.legend(loc=legend_loc, prop=prop)

        plt.tight_layout()


def plot_class_histograms(data, columns, class_column='CLASS', title=None, log_y=False):
    color_palette = get_cubehelix_palette(len(BASE_CLASSES))
    for c in columns:
        plt.figure()
        for i, t in enumerate(BASE_CLASSES):
            sns.distplot(data.loc[data[class_column] == t][c], label=t, kde=False, rug=False, color=color_palette[i],
                         hist_kws={'alpha': 1.0, 'histtype': 'step', 'linewidth': 1.5, 'linestyle': get_line_style(i)})
        if log_y: plt.yscale('log')
        if title: plt.title(title)
        plt.xlabel(pretty_print_feature(c))
        plt.legend()


def plot_map(hpxmap, unit='counts per pixel', is_cmap=True):
    cmap = plt.get_cmap('hot_r') if is_cmap else None
    hp.mollzoom(hpxmap, cmap=cmap, nest=False, xsize=1600, unit=unit, title='Galactic coordinates')
    hp.graticule()


def plot_non_zero_map_stats(m, lat, map_stars=None, title=None):
    i_non_zero = np.nonzero(m)
    m_to_plot = m[i_non_zero]
    lat_to_plot = lat[i_non_zero]
    map_stars_to_plot = map_stars[i_non_zero] if map_stars is not None else None

    plot_map_stats(m_to_plot, lat_to_plot, map_stars=map_stars_to_plot, title=title)


def plot_map_stats(m, lat, map_stars=None, title=None, xlim=None):
    bins = 50

    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(lat, m, statistic='mean', bins=bins)
    # plt.bar(bin_edges[:-1], bin_means, bin_edges[1] - bin_edges[0], align='edge')
    # plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:])
    plt.plot(bin_edges[:-1], bin_means)
    plt.xlim(xlim)
    plt.xlabel('galactic latitude')
    plt.ylabel('mean pixel density')
    plt.title(title)

    if map_stars is not None:
        plt.figure()
        bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(lat, map_stars, statistic='mean', bins=bins)
        # plt.bar(bin_edges[:-1], bin_means, bin_edges[1] - bin_edges[0], align='edge')
        # plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:])
        plt.plot(bin_edges[:-1], bin_means, label='stars')
        plt.xlim(xlim)
        plt.xlabel('galactic latitude')
        plt.ylabel('mean pixel density')
        plt.title('stars')

    plt.figure()
    sns.distplot(m)
    plt.xlabel('pixel density')
    plt.ylabel('counts')
    plt.title(title)


def plot_proba_against_size(data, column='QSO', x_lim=(0, 1), step=0.01):
    thresholds = np.arange(x_lim[0], x_lim[1] + step, step)
    data_size_arr = [data.loc[data[column] >= thr].shape[0] for thr in thresholds]

    plt.plot(thresholds, data_size_arr)
    plt.xlabel('{} probability threshold'.format(column))
    plt.ylabel('{} size'.format(column))


def plot_external_qso_consistency(catalog):
    step = 0.05
    thresholds = np.arange(0.3, 1.0, step)
    color_palette = get_cubehelix_palette(len(EXTERNAL_QSO_DICT))

    # Read data
    data_dict = OrderedDict(
        (data_name, pd.read_csv(data_path)) for data_name, data_path in EXTERNAL_QSO_DICT.items())

    # Take only QSOs for 2QZ/6QZ
    data_tmp = process_2df(data_dict['x 2QZ/6QZ'])
    data_dict['x 2QZ/6QZ'] = data_tmp.loc[data_tmp['id1'] == 'QSO']

    plt.figure()

    for i, (external_qso_name, external_qso) in enumerate(data_dict.items()):
        threshold_data_arr = [catalog.loc[catalog[['QSO', 'STAR', 'GALAXY']].max(axis=1) >= thr][['ID', 'CLASS']] for
                              thr in
                              thresholds]
        intersection_data_arr = [thr_data.loc[thr_data['ID'].isin(external_qso['ID'])] for thr_data in
                                 threshold_data_arr]
        qso_data_arr = [int_data.loc[int_data['CLASS'] == 'QSO'] for int_data in intersection_data_arr]

        agreement_arr = [qso_data_arr[i].shape[0] / float(intersection_data_arr[i].shape[0]) for i, _ in
                         enumerate(qso_data_arr)]

        # TODO: Ugly work around
        external_qso_name = get_external_qso_short_name(external_qso_name)
        # if external_qso_name == 'x 6QZ':
        external_qso_name += ' QSO'

        plt.plot(thresholds, agreement_arr, label=external_qso_name, linestyle=get_line_style(i), alpha=1.0,
                 color=color_palette[i])

    plt.xlabel('KiDS minimum probability')
    plt.ylabel('KiDS QSO contribution')
    legend = plt.legend(loc='lower left')
    plt.tight_layout()

    ax = plt.axes()
    ax.yaxis.grid(True)

    # Customized legend place
    ax = plt.gca()
    bounding_box = legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    y_offset = 0.08
    bounding_box.y0 += y_offset
    bounding_box.y1 += y_offset
    legend.set_bbox_to_anchor(bounding_box, transform=ax.transAxes)


def plot_external_qso_size(data):
    step = 0.01
    thresholds = np.arange(0.3, 1.0 + step, step)
    color_palette = get_cubehelix_palette(len(EXTERNAL_QSO_DICT))

    # Read data
    data_dict = OrderedDict(
        (data_name, pd.read_csv(data_path)) for data_name, data_path in EXTERNAL_QSO_DICT.items())

    # Take only QSOs for 2QZ/6QZ
    data_tmp = process_2df(data_dict['x 2QZ/6QZ'])
    data_dict['x 2QZ/6QZ'] = data_tmp.loc[data_tmp['id1'] == 'QSO']

    for i, (external_qso_name, external_qso) in enumerate(data_dict.items()):
        data_size_arr = [sum(data.loc[data['QSO'] >= thr]['ID'].isin(external_qso['ID'])) for thr in thresholds]

        # TODO: Ugly work around
        if external_qso_name == 'x 2QZ/6QZ':
            external_qso_name += ' quasars'

        plt.plot(thresholds, data_size_arr, label=external_qso_name, linestyle=get_line_style(i), alpha=1.0,
                 color=color_palette[i])
        plt.xlabel('QSO probability threshold')
        plt.ylabel('cross match size')

    plt.legend()


def plot_feature_ranking(model, features):
    importances = model.feature_importances_ * 100
    # no std because it's too big
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    max_features = 40
    if len(features) > max_features:
        indices = indices[:max_features]

    features_sorted = np.array(features)[indices]
    importances_sorted = np.array(importances)[indices]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(features_sorted)), importances_sorted, align='center', color=get_cubehelix_palette(1)[0])
    ax.set_yticks(range(len(features_sorted)))

    feature_names = [pretty_print_feature(feature_name) for feature_name in np.array(features_sorted)]
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('feature importance (%)')

    for i, value in enumerate(importances_sorted):
        offset = -5.0 if i == 0 else .35
        color = 'white' if i == 0 else 'black'
        ax.text(value + offset, i + .2, '{:.2f}%'.format(value), color=color)

    plt.tight_layout()
    plt.show()