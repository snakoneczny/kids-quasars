import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embedding(embedding, labels, label_name='class'):
    labels_unique = np.unique(labels)
    n_colors = len(labels_unique)
    color_palette = sns.color_palette('muted', n_colors)
    color_palette = {labels_unique[k]: v for k, v in enumerate(color_palette)}
    color_palette['UNKNOWN'] = (0.6, 0.6, 0.6)

    data = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], label_name: labels})
    sns.lmplot(x='x', y='y', hue=label_name, data=data, palette=color_palette, fit_reg=False, scatter_kws={'alpha': 0.5},
               size=7)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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


def plot_class_histograms(data, columns):
    for c in columns:
        plt.figure()
        for t in ['STAR', 'GALAXY', 'QSO']:
            sns.distplot(data.loc[data['CLASS'] == t][c], label=t, kde=False, rug=False, hist_kws={'alpha': 0.5})
        plt.legend()
