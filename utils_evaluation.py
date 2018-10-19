import os
from collections.__init__ import defaultdict

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, roc_curve, auc, precision_score, \
    recall_score, average_precision_score, precision_recall_curve

from utils import *
from utils_plotting import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, get_line_style, \
    get_cubehelix_palette


def classification_report(predictions, col_true='CLASS', z_max=None):
    class_names = np.unique(predictions[col_true])
    predictions['class_pred'] = predictions[class_names].idxmax(axis=1)

    np.set_printoptions(precision=4)

    multiclass_report(predictions, class_names, col_true=col_true)
    binary_report(predictions, col_true=col_true)

    if 'Z' in predictions.columns:
        classification_z_report(predictions, col_true=col_true, z_max=z_max)


def multiclass_report(predictions, class_names, col_true='CLASS'):
    print('Multiclass classification results:')

    y_true = predictions[col_true]
    y_pred = predictions['class_pred']

    acc = accuracy_score(y_true, y_pred)
    print('Accuracy = {:.4f}'.format(acc))

    f1 = f1_score(y_true, y_pred, average=None)
    print('F1 per class = {}'.format(f1))

    logloss = log_loss(y_true, predictions[['GALAXY', 'QSO', 'STAR']])
    print('Logloss = {:.4f}'.format(logloss))

    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()


def binary_report(predictions, col_true='CLASS'):
    print('Binary classification results:')

    y_true = (predictions[col_true] == 'QSO')
    y_pred_proba = predictions['QSO']
    y_pred_binary = (predictions['class_pred'] == 'QSO')

    n_pos = y_pred_binary.sum()
    n_all = len(y_pred_binary)
    print('Predicted positives: {}/{} ({:.2f}%)'.format(n_pos, n_all, n_pos / n_all * 100))

    logloss = log_loss(y_true, y_pred_proba)
    print('logloss = {:.4f}'.format(logloss))

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print('ROC AUC = {:.4f}'.format(roc_auc))
    plot_roc_curve(fpr, tpr, roc_auc)

    binary_metrics = OrderedDict([
        ('accuracy', accuracy_score),
        ('f1', f1_score),
        ('precision', precision_score),
        ('recall', recall_score),
    ])
    for metric_name, metric_func in binary_metrics.items():
        metric_value = metric_func(y_true, y_pred_binary)
        print('{} = {:.4f}'.format(metric_name, metric_value))

    # Precision - recall curve
    average_precision = average_precision_score(y_true, y_pred_proba)
    precision, recall = precision_score(y_true, y_pred_binary), recall_score(y_true, y_pred_binary)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(precisions, recalls, average_precision, precision, recall)


def classification_z_report(predictions, col_true='CLASS', z_max=None):
    predictions_zlim = predictions.loc[predictions['Z'] <= z_max]

    for class_true in BASE_CLASSES:

        true_class_as_dict = {}
        for class_pred in BASE_CLASSES:
            true_class_as_dict[class_pred] = predictions_zlim.loc[
                (predictions_zlim[col_true] == class_true) & (predictions_zlim['class_pred'] == class_pred)]['Z']

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


def redshift_report(predictions):
    predictions['residual'] = abs(predictions['Z'] - predictions['Z_pred'])
    bins = np.arange(predictions['Z'].min(), predictions['Z'].max() + 0.5, 0.5)
    predictions['binned'] = pd.cut(predictions['Z'], bins)

    grouped = predictions.groupby(by='binned')

    print(predictions['residual'].mean())
    print(grouped.size())
    print(grouped.mean()['residual'])


def metric_class_split(y_true, y_pred, classes, metric):
    scores = []
    for c in np.unique(classes):
        c_idx = np.array((classes == c))
        scores.append(metric(y_true[c_idx], y_pred[c_idx]))
    return scores


def number_counts(data, x_lim=None, title=None, legend_loc='upper left'):
    # Get x limit from all magnitudes
    if x_lim is None:
        m_min = int(math.floor(data[BAND_CALIB_COLUMNS].values.min()))
        m_max = int(math.ceil(data[BAND_CALIB_COLUMNS].values.max()))
        bins = np.arange(m_min, m_max + 1.0, 1.0)
    else:
        bins = np.arange(x_lim[0], x_lim[1] + 1.0, 1.0)

    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    # Plot for every magnitude
    counts = pd.DataFrame()
    for band in BAND_CALIB_COLUMNS:

        # Bin magnitudes
        data['bin'] = pd.cut(data[band], bins, labels=False)

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


def number_counts_multidata(data_dict, x_lim, band=MAG_GAAP_CALIB_R, legend_loc='upper left'):
    bins = np.arange(x_lim[0], x_lim[1] + .5, .5)
    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    counts = pd.DataFrame()
    for data_name, data in data_dict.items():
        data['bin'] = pd.cut(data[band], bins, labels=False)

        # For each bin
        for i in range(len(bins) - 1):
            data_bin = data.loc[data['bin'] == i]
            counts = counts.append({'objects': data_bin.shape[0], band: bin_titles[i], 'dataset': data_name},
                                   ignore_index=True)

    sns.catplot(x=band, y='objects', hue='dataset', data=counts, kind='bar',
                aspect=1.6, height=5, legend_out=False, palette='cubehelix')
    plt.legend(loc=legend_loc)
    plt.xlabel(pretty_print_magnitude(band))
    plt.xticks(rotation=30)
    plt.ylabel('counts per bin')
    plt.yscale('log')


def number_counts_linear(data, c=10, linear_range=(18, 20)):
    for b in BAND_CALIB_COLUMNS:

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
def number_counts_pixels(data, nside=58, x_lim=None, title=None, legend_loc='upper left'):
    # Get mask for the whole dataset
    map, _, _ = get_map(data['RAJ2000'], data['DECJ2000'], nside=nside)
    mask_non_zero = np.nonzero(map)

    # Get x limit from all magnitudes
    if x_lim is None:
        m_min = int(math.floor(data[BAND_CALIB_COLUMNS].values.min()))
        m_max = int(math.ceil(data[BAND_CALIB_COLUMNS].values.max()))
        bins = np.arange(m_min, m_max + 1.0, 1.0)
    else:
        bins = np.arange(x_lim[0], x_lim[1] + 1.0, 1.0)

    bin_titles = ['({}, {}]'.format(bins[i], bins[i + 1]) for i, _ in enumerate(bins[:-1])]

    # Plot for every magnitude
    pixel_densities = pd.DataFrame()
    for band in BAND_CALIB_COLUMNS:

        # Bin magnitudes
        data['bin'] = pd.cut(data[band], bins, labels=False)

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


def test_external_qso(catalog, save=False):
    print('catalog size: {}'.format(catalog.shape[0]))
    print(describe_column(catalog['CLASS']))

    for external_path in EXTERNAL_QSO_PATHS:
        external_catalog = pd.read_csv(external_path)

        # Take only QSOs for 2QZ/6QZ
        if 'id1' in external_catalog.columns:
            external_catalog = process_2df(external_catalog)
            external_catalog = external_catalog.loc[external_catalog['id1'] == 'QSO']

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
                             hist_kws={'alpha': 0.5, 'histtype': 'step'})
                plt.title(title)
            plt.legend()

    if save:
        catalog.loc[is_in_ext].to_csv('catalogs_intersection/{}.csv'.format(title))


def gaia_motion_analysis(data, norm=False):
    movement_mask = ~data[['parallax', 'pmdec', 'pmra']].isnull().any(axis=1)
    data_movement = data.loc[movement_mask]

    classes = ['QSO', 'GALAXY', 'STAR']
    for class_name in classes:

        motions = ['parallax', 'pmra', 'pmdec']
        if norm & (class_name == 'QSO'):
            motions = [m + '_norm' for m in motions]
        result_df = pd.DataFrame(index=['mu', 'median', 'sigma'], columns=motions)

        for motion in motions:
            data_of_interest = data_movement.loc[data_movement['CLASS'] == class_name, motion]
            (mu, sigma) = stats.norm.fit(data_of_interest)
            median = np.median(data_of_interest)

            result_df.loc['mean', motion] = mu
            result_df.loc['median', motion] = median
            result_df.loc['sigma', motion] = sigma

            plt.figure()
            sns.distplot(data_of_interest, color=get_cubehelix_palette(1)[0], kde_kws=dict(bw=0.5))
            # if motion == 'parallax_norm':
            #     plt.xlim((-6, 6))
            plt.ylabel(class_name)

        print('{}:'.format(class_name))
        display(result_df)


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
            plt.ylabel(t[1])

        plt.legend()
