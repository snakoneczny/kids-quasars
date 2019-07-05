import os
from collections.__init__ import defaultdict, OrderedDict
from functools import partial

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, roc_curve, auc, precision_score, \
    recall_score, average_precision_score, precision_recall_curve, mean_absolute_error, mean_squared_error

from data import EXTERNAL_QSO_PATHS, BASE_CLASSES, BAND_COLUMNS, get_mag_str, clean_gaia, process_2df
from utils import *
from plotting import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, get_line_style, \
    get_cubehelix_palette, plot_proba_histograms


def experiment_report(predictions, z_max=None, col_true='CLASS', true_label='SDSS'):
    np.set_printoptions(precision=4)

    if 'CLASS_PHOTO' in predictions.columns:
        multiclass_report(predictions, col_true=col_true, true_label=true_label)
        binary_report(predictions, col_true=col_true)

        if 'Z' in predictions.columns:
            classification_completeness_z_report(predictions, col_true=col_true, z_max=z_max)

    if 'Z_PHOTO' in predictions.columns:
        redshift_report(predictions, z_max=z_max)

    if 'CLASS_PHOTO' in predictions.columns and 'Z_PHOTO' in predictions.columns:
        classification_precision_z_report(predictions, z_max=z_max)
        classification_and_redshift_report(predictions)


def multiclass_report(predictions, col_true='CLASS', true_label='SDSS'):
    class_names = np.unique(predictions[col_true])

    print('Multiclass classification results:')

    y_true = predictions[col_true]
    y_pred = predictions['CLASS_PHOTO']

    acc = accuracy_score(y_true, y_pred)
    print('Accuracy = {:.4f}'.format(acc))

    f1 = f1_score(y_true, y_pred, average=None)
    print('F1 per class = {}'.format(f1))

    logloss = log_loss(y_true, predictions[['GALAXY_PHOTO', 'QSO_PHOTO', 'STAR_PHOTO']])
    print('Logloss = {:.4f}'.format(logloss))

    # Confusion matrices
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, true_label=true_label)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, true_label=true_label)
    plt.show()

    plot_proba_histograms(predictions)


def binary_report(predictions, col_true='CLASS'):
    print('Binary classification results:')

    y_true = (predictions[col_true] == 'QSO')
    y_pred_proba = predictions['QSO_PHOTO']
    y_pred_binary = (predictions['CLASS_PHOTO'] == 'QSO')

    n_pos = y_pred_binary.sum()
    n_all = len(y_pred_binary)
    print('Predicted positives: {}/{} ({:.2f}%)'.format(n_pos, n_all, n_pos / n_all * 100))

    logloss = log_loss(y_true, y_pred_proba)
    print('Logloss = {:.4f}'.format(logloss))

    binary_metrics = OrderedDict([
        ('Accuracy', accuracy_score),
        ('F1', f1_score),
        ('Precision', precision_score),
        ('Recall', recall_score),
    ])
    for metric_name, metric_func in binary_metrics.items():
        metric_value = metric_func(y_true, y_pred_binary)
        print('{} = {:.4f}'.format(metric_name, metric_value))

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print('ROC AUC = {:.4f}'.format(roc_auc))
    plot_roc_curve(fpr, tpr, roc_auc)

    # Precision - recall curve
    average_precision = average_precision_score(y_true, y_pred_proba)
    precision, recall = precision_score(y_true, y_pred_binary), recall_score(y_true, y_pred_binary)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(precisions, recalls, average_precision, precision, recall)


def classification_completeness_z_report(predictions, col_true='CLASS', z_max=None):
    """
    Compare predicted classes against true redshifts
    :param predictions:
    :param col_true:
    :param z_max:
    :return:
    """
    predictions_zlim = predictions.loc[predictions['Z'] <= z_max]

    for class_true in BASE_CLASSES:

        true_class_as_dict = {}
        for class_pred in BASE_CLASSES:
            true_class_as_dict[class_pred] = predictions_zlim.loc[
                (predictions_zlim[col_true] == class_true) & (predictions_zlim['CLASS_PHOTO'] == class_pred)]['Z']

        plt.figure()
        _, bin_edges = np.histogram(
            np.hstack((true_class_as_dict['QSO'], true_class_as_dict['STAR'], true_class_as_dict['GALAXY'])), bins=40)
        color_palette = get_cubehelix_palette(len(BASE_CLASSES))

        for i, class_pred in enumerate(BASE_CLASSES):
            sns.distplot(true_class_as_dict[class_pred], label='{} clf. as {}'.format(class_true, class_pred),
                         bins=bin_edges, kde=False, rug=False, color=color_palette[i],
                         hist_kws={'alpha': 1.0, 'histtype': 'step', 'linewidth': 1.5, 'linestyle': get_line_style(i)})

        plt.xlabel('redshift')
        plt.ylabel('counts per bin')
        plt.legend(loc='upper left')


def relative_error(y_true, y_pred):
    e = (y_pred - y_true) / (1 + y_true)
    return e.mean()


def relative_error_std(y_true, y_pred):
    e = (y_pred - y_true) / (1 + y_true)
    return e.std()


# TODO: Refactor
def redshift_report(predictions, z_max=None):
    classes = np.unique(predictions['CLASS'])

    # Standard metrics
    metrics_redshift = [('MSE', mean_squared_error), ('MAE', mean_absolute_error),
                        ('rel. error', relative_error), ('rel. error std', relative_error_std)]
    for metric_name, metric_func in metrics_redshift:
        score = np.around(metric_func(predictions['Z'], predictions['Z_PHOTO']), 4)

        print('{metric_name}: {score}'.format(metric_name=metric_name, score=score))

        # Divided for classes
        scores = np.around(metric_class_split(predictions['Z'], predictions['Z_PHOTO'], metric=metric_func,
                                              classes=predictions['CLASS']), 4)
        print(', '.join(['{c}: {s}'.format(metric_name=metric_name, c=c, s=s) for c, s in zip(classes, scores)]))

        # Divided for photometric classes
        if 'CLASS_PHOTO' in predictions:
            scores = np.around(metric_class_split(predictions['Z'], predictions['Z_PHOTO'], metric=metric_func,
                                                  classes=predictions['CLASS_PHOTO']), 4)
            print(', '.join(
                ['{c} photo: {s}'.format(metric_name=metric_name, c=c, s=s) for c, s in zip(classes, scores)]))

    # Show binned statistics
    redshifts_per_class_dict = {}
    for c in classes:
        redshifts_per_class_dict[c] = predictions.loc[predictions['CLASS'] == c]['Z']

    _, bin_edges = np.histogram(np.hstack(
        (redshifts_per_class_dict['QSO'], redshifts_per_class_dict['STAR'], redshifts_per_class_dict['GALAXY'])),
        bins=40)

    # Calculate errors in redshift bins
    predictions.loc[:, 'binned'] = pd.cut(predictions['Z'], bin_edges)
    predictions.loc[:, 'residual'] = abs(predictions['Z'] - predictions['Z_PHOTO'])
    predictions.loc[:, 'residual_sqr'] = (predictions['Z'] - predictions['Z_PHOTO']) ** 2

    mae_dict, mse_dict, size_dict = {}, {}, {}
    for i, c in enumerate(classes):
        preds_class = predictions.loc[predictions['CLASS'] == c]
        grouped = preds_class.groupby(by='binned')

        mae_dict[c] = grouped['residual'].mean().values
        mse_dict[c] = grouped['residual_sqr'].mean().values
        size_dict[c] = grouped.size().values

    # Plot errors in bins of predicted redshift
    to_plot = [(mae_dict, 'mean absolute error'), (mse_dict, 'mean square error'), (size_dict, 'number of objects')]
    color_palette = get_cubehelix_palette(len(classes))
    for x_dict, plot_title in to_plot:
        plt.figure()
        for i, c in enumerate(classes):
            ax = sns.lineplot(bin_edges[:-1], x_dict[c], drawstyle='steps-pre', label=c, color=color_palette[i])
            ax.lines[i].set_linestyle(get_line_style(i))
        plt.xlabel('redshift')
        plt.ylabel(plot_title)
        plt.legend()

    # Plot true vs predicted redshifts
    plot_z_true_vs_pred(predictions, 'Z_PHOTO', z_max)
    # Plot Z_B for comparison
    plot_z_true_vs_pred(predictions, 'Z_B', z_max)


def plot_z_true_vs_pred(predictions, z_col, z_max):
    z_max = {'GALAXY': min(1, z_max), 'QSO': z_max}
    for c in ['GALAXY', 'QSO']:
        preds_c = predictions.loc[predictions['CLASS'] == c]

        plt.figure()
        p = sns.scatterplot(x='Z', y=z_col, data=preds_c)
        plt.plot(range(z_max[c] + 1))
        p.set(xlim=(0, z_max[c]), ylim=(0, z_max[c]))
        plt.title(c)

        plt.figure()
        p = sns.kdeplot(preds_c['Z'], preds_c[z_col], shade=True)
        plt.plot(range(z_max[c] + 1))
        p.set(xlim=(0, z_max[c]), ylim=(0, z_max[c]))
        plt.title(c)


# TODO: refactor, its doubled in classification_precision_z_report report
def plot_z_hist(predictions, z_max=None):
    predictions_zlim = predictions.loc[predictions['Z'] <= z_max]

    n_bins = 40
    # _, bin_edges = np.histogram(predictions_zlim['Z_PHOTO'], bins=n_bins)
    predictions_zlim.loc[:, 'binned'] = pd.cut(predictions_zlim['Z'], n_bins)

    counts_dict = {}  # final index: [predicted class][true class]
    for i, class_pred in enumerate(BASE_CLASSES):
        preds_class = predictions_zlim.loc[predictions_zlim['CLASS_PHOTO'] == class_pred]
        counts_dict[class_pred] = preds_class.groupby(['binned', 'CLASS']).size().unstack(fill_value=0)
        counts_dict[class_pred].loc[:, 'size'] = counts_dict[class_pred][BASE_CLASSES].sum(axis=1)

    # Size plot
    plt.figure()

    color_palette = get_cubehelix_palette(len(BASE_CLASSES))
    for i, class_pred in enumerate(BASE_CLASSES):
        bin_edges_left = [v.left for v in counts_dict[class_pred].index]
        size_list_norm = counts_dict[class_pred]['size'] / max(counts_dict[class_pred]['size'])
        ax = sns.lineplot(bin_edges_left, size_list_norm, drawstyle='steps-pre', label=class_pred,
                          color=color_palette[i])
        ax.lines[i].set_linestyle(get_line_style(i))

    plt.xlabel('z')
    plt.ylabel('number of objects in bin')
    plt.legend()


def classification_precision_z_report(predictions, z_max=None):
    """
    Compare precision against predicted redshift
    :param predictions:
    :param z_max:
    :return:
    """
    predictions_zlim = predictions.loc[predictions['Z'] <= z_max]

    n_bins = 40
    # _, bin_edges = np.histogram(predictions_zlim['Z_PHOTO'], bins=n_bins)
    predictions_zlim.loc[:, 'binned'] = pd.cut(predictions_zlim['Z_PHOTO'], n_bins)

    counts_dict = {}  # final index: [predicted class][true class]
    for i, class_pred in enumerate(BASE_CLASSES):
        preds_class = predictions_zlim.loc[predictions_zlim['CLASS_PHOTO'] == class_pred]
        counts_dict[class_pred] = preds_class.groupby(['binned', 'CLASS']).size().unstack(fill_value=0)
        counts_dict[class_pred].loc[:, 'size'] = counts_dict[class_pred][BASE_CLASSES].sum(axis=1)

    # Precision plot
    plt.figure()
    color_palette = get_cubehelix_palette(len(BASE_CLASSES))
    for i, c in enumerate(BASE_CLASSES):
        bin_edges_left = [v.left for v in counts_dict[c].index]
        precision_list = counts_dict[c][c] / counts_dict[c]['size']
        ax = sns.lineplot(bin_edges_left, precision_list, drawstyle='steps-pre', label=c, color=color_palette[i])
        ax.lines[i].set_linestyle(get_line_style(i))

    plt.xlabel('z photo')
    plt.ylabel('purity')
    plt.legend()

    # True z size plot for comparison
    plot_z_hist(predictions, z_max=z_max)

    # Size plot
    plt.figure()
    for i, class_pred in enumerate(BASE_CLASSES):
        bin_edges_left = [v.left for v in counts_dict[class_pred].index]
        size_list_norm = counts_dict[class_pred]['size'] / max(counts_dict[class_pred]['size'])
        ax = sns.lineplot(bin_edges_left, size_list_norm, drawstyle='steps-pre', label=class_pred,
                          color=color_palette[i])
        ax.lines[i].set_linestyle(get_line_style(i))

    plt.xlabel('z photo')
    plt.ylabel('number of predictions in bin')
    plt.legend()

    # Detailed plots
    for class_pred in BASE_CLASSES:
        bin_edges_left = [v.left for v in counts_dict[class_pred].index]

        plt.figure()
        color_palette = get_cubehelix_palette(len(BASE_CLASSES))
        for i, class_true in enumerate(BASE_CLASSES):
            counts_per_class = counts_dict[class_pred][class_true] / counts_dict[class_pred]['size']

            ax = sns.lineplot(bin_edges_left, counts_per_class, drawstyle='steps-pre', label=class_true,
                              color=color_palette[i])
            ax.lines[i].set_linestyle(get_line_style(i))

        plt.title('{} photo'.format(class_pred))
        plt.xlabel('z photo')
        plt.ylabel('purity')
        plt.legend()


def classification_and_redshift_report(predictions):
    step = 0.02
    thresholds = np.arange(0, 1, step)
    classes = np.unique(predictions['CLASS'])
    color_palette = get_cubehelix_palette(len(classes))

    # True class plots
    metrics_to_plot = OrderedDict([('MAE', mean_absolute_error), ('MSE', mean_squared_error)])
    for metric_name, metric_func in metrics_to_plot.items():

        plt.figure()
        for i, c in enumerate(classes):
            preds_class = predictions.loc[predictions['CLASS'] == c]

            # Get scores limited by classification probability thresholds
            metric_values = []
            for thr in thresholds:
                preds_lim = preds_class.loc[preds_class['{}_PHOTO'.format(c)] >= thr]
                metric_values.append(np.around(metric_func(preds_lim['Z'], preds_lim['Z_PHOTO']), 4))

            plt.plot(thresholds, metric_values, label='true {}'.format(c), color=color_palette[i],
                     linestyle=get_line_style(i))
            plt.xlabel('minimum classification probability')
            plt.ylabel(metric_name)
            ax = plt.axes()
            ax.yaxis.grid(True)

        plt.legend()

    # TODO: refactor
    # True class plots
    metrics_to_plot = OrderedDict([('MSE', mean_absolute_error), ('MAE', mean_squared_error)])
    for metric_name, metric_func in metrics_to_plot.items():

        plt.figure()
        for i, c in enumerate(classes):
            preds_class = predictions.loc[predictions['CLASS_PHOTO'] == c]

            # Get scores limited by classification probability thresholds
            metric_values = []
            for thr in thresholds:
                preds_lim = preds_class.loc[preds_class['{}_PHOTO'.format(c)] >= thr]
                metric_values.append(np.around(metric_func(preds_lim['Z'], preds_lim['Z_PHOTO']), 4))

            plt.plot(thresholds, metric_values, label='photo {}'.format(c), color=color_palette[i],
                     linestyle=get_line_style(i))
            plt.xlabel('minimum classification probability')
            plt.ylabel(metric_name)
            ax = plt.axes()
            ax.yaxis.grid(True)

        plt.legend()


def metric_class_split(y_true, y_pred, classes, metric):
    scores = []
    for c in np.unique(classes):
        c_idx = np.array((classes == c))
        scores.append(metric(y_true[c_idx], y_pred[c_idx]))
    return scores


def number_counts(data, x_lim=None, title=None, legend_loc='upper left', columns=BAND_COLUMNS):
    # Get x limit from all magnitudes
    if x_lim is None:
        m_min = int(math.floor(data[columns].values.min()))
        m_max = int(math.ceil(data[columns].values.max()))
        bins = np.arange(m_min, m_max + 1.0, 1.0)
    else:
        bins = np.arange(x_lim[0], x_lim[1] + 1.0, 1.0)

    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    # Plot for every magnitude
    counts = pd.DataFrame()
    for band in columns:

        # Bin magnitudes
        data.loc[:, 'bin'] = pd.cut(data[band], bins, labels=False)

        # For each bin
        for i in range(len(bins) - 1):
            data_bin = data.loc[data['bin'] == i]
            counts = counts.append({'objects': data_bin.shape[0], 'magnitude range': bin_titles[i], 'magnitude': band},
                                   ignore_index=True)

    sns.catplot(x='magnitude range', y='objects', hue='magnitude', data=counts, kind='bar',
                aspect=1.7, height=5, legend_out=False, palette='cubehelix')
    plt.legend(loc=legend_loc)
    plt.yscale('log')
    plt.title(title)


def number_counts_multidata(data_dict, x_lim, band='r', legend_loc='upper left'):
    band_column = get_mag_str(band)
    bins = np.arange(x_lim[0], x_lim[1] + .5, .5)
    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    counts = pd.DataFrame()
    for data_name, data in data_dict.items():
        data.loc[:, 'bin'] = pd.cut(data[band_column], bins, labels=False)

        # For each bin
        for i in range(len(bins) - 1):
            data_bin = data.loc[data['bin'] == i]
            counts = counts.append({'objects': data_bin.shape[0], band_column: bin_titles[i], 'dataset': data_name},
                                   ignore_index=True)

    sns.catplot(x=band, y='objects', hue='dataset', data=counts, kind='bar',
                aspect=1.6, height=5, legend_out=False, palette='cubehelix')
    plt.legend(loc=legend_loc)
    plt.xlabel(pretty_print_magnitude(band))
    plt.xticks(rotation=30)
    plt.ylabel('counts per bin')
    plt.yscale('log')


def number_counts_linear(data, c=10, linear_range=(18, 20), columns=BAND_COLUMNS):
    for b in columns:

        m_min = int(math.ceil(data[b].min()))
        m_max = int(math.ceil(data[b].max()))

        x, y = [], []
        for m in range(m_min, m_max + 1):
            x.append(m)
            v = data.loc[data[b] < m].shape[0]
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


# nside 58 gives 1.02 sq. deg.
def number_counts_pixels(data, nside=58, x_lim=None, title=None, legend_loc='upper left', columns=BAND_COLUMNS):
    # Get mask for the whole dataset
    map, _, _ = get_map(data['RAJ2000'], data['DECJ2000'], nside=nside)
    mask_non_zero = np.nonzero(map)

    # Get x limit from all magnitudes
    if x_lim is None:
        m_min = int(math.floor(data[columns].values.min()))
        m_max = int(math.ceil(data[columns].values.max()))
        bins = np.arange(m_min, m_max + 1.0, 1.0)
    else:
        bins = np.arange(x_lim[0], x_lim[1] + 1.0, 1.0)

    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    # Plot for every magnitude
    pixel_densities = pd.DataFrame()
    for band in columns:

        # Bin magnitudes
        data.loc[:, 'bin'] = pd.cut(data[band], bins, labels=False)

        # For each bin
        for i in range(len(bins) - 1):
            data_bin = data.loc[data['bin'] == i]

            # Get map
            map, _, _ = get_map(data_bin['RAJ2000'], data_bin['DECJ2000'], nside=nside)
            map_masked = map[mask_non_zero]
            pixel_densities = pixel_densities.append(
                pd.DataFrame({'pixel density': map_masked, 'magnitude range': bin_titles[i], 'magnitude': band}),
                ignore_index=True)

    sns.catplot(x='magnitude range', y='pixel density', hue='magnitude', data=pixel_densities, kind='bar',
                aspect=1.7, height=5, legend_out=False, palette='cubehelix')
    plt.legend(loc=legend_loc)
    plt.yscale('log')
    plt.title(title)


def test_external_qso(catalog, save=False, plot=True):
    print('catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog['CLASS']))

    for external_path in EXTERNAL_QSO_PATHS:
        external_catalog = pd.read_csv(external_path)

        # Take only QSOs for 2QZ/6QZ
        if 'id1' in external_catalog.columns:
            external_catalog = process_2df(external_catalog)
            external_catalog = external_catalog.loc[external_catalog['id1'] == 'QSO']

        title = os.path.basename(external_path)[:-4]
        test_against_external_catalog(external_catalog, catalog, title=title, plot=plot, save=save)


def test_gaia(catalog, catalog_x_gaia_path, class_column='CLASS', id_column='ID', save=False):
    print('catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog[class_column]))

    catalog_x_gaia = pd.read_csv(catalog_x_gaia_path)

    movement_mask = ~catalog_x_gaia[['parallax', 'pmdec', 'pmra']].isnull().any(axis=1)
    catalog_x_gaia_movement = catalog_x_gaia.loc[movement_mask]

    catalog_x_gaia_movement = clean_gaia(catalog_x_gaia_movement)

    test_against_external_catalog(catalog_x_gaia_movement, catalog, class_column=class_column, id_column=id_column,
                                  title='GAIA', save=save)


def test_against_external_catalog(ext_catalog, catalog, columns=BAND_COLUMNS, class_column='CLASS', id_column='ID',
                                  title='', plot=True, save=False):
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
    if plot:
        if 'MAG_GAAP_CALIB_U' in catalogs_cross.columns:
            for c in columns:
                plt.figure()
                for t in BASE_CLASSES:
                    sns.distplot(catalogs_cross.loc[catalogs_cross['CLASS'] == t][c], label=t, kde=False, rug=False,
                                 hist_kws={'alpha': 0.5, 'histtype': 'step'})
                    plt.title(title)
                plt.legend()

    if save:
        catalog.loc[is_in_ext].to_csv('catalogs_intersection/{}.csv'.format(title))


def gaia_motion_analysis(data, norm=False):
    movement_mask = ~data[['parallax', 'pmdec', 'pmra']].isnull().any(axis=1)
    data_movement = data.loc[movement_mask]

    for class_name in BASE_CLASSES:

        motions = ['parallax', 'pmra', 'pmdec']
        if norm & (class_name == 'QSO'):
            motions = [m + '_norm' for m in motions]
        result_df = pd.DataFrame(index=['mean', 'median', 'sigma'], columns=motions)

        for motion in motions:
            data_of_interest = data_movement.loc[data_movement['CLASS'] == class_name, motion]
            (mu, sigma) = stats.norm.fit(data_of_interest)
            median = np.median(data_of_interest)

            result_df.loc['mean', motion] = mu
            result_df.loc['median', motion] = median
            result_df.loc['sigma', motion] = sigma

            plt.figure()
            sns.distplot(data_of_interest, color=get_cubehelix_palette(1)[0], kde_kws=dict(bw=0.5))
            if motion == 'parallax':
                plt.xlim((-6, 6))
            plt.ylabel(class_name)

        print('{}:'.format(class_name))
        print(result_df)


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
    to_plot = [(mu_dict, 'mean'), (sigma_dict, 'sigma'), (median_dict, 'median')]
    color_palette = get_cubehelix_palette(len(motions))

    for t in to_plot:
        plt.figure()

        for i, motion in enumerate(motions):
            plt.plot(thresholds, t[0][motion], label=motion, color=color_palette[i], linestyle=get_line_style(i))
            plt.xlabel('minimum classification probability')
            plt.ylabel('{} {}'.format(t[1], '[mas]'))

            ax = plt.axes()
            ax.yaxis.grid(True)

        plt.legend()
