##The credit for this module goes to http://scikit-learn.org/stable/about.html#citing-scikit-learn
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_pred=None, y_true=None, classes=None, normalize=False, cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    title = 'Confusion matrix'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

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
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_score(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


def get_prf1a(tp, fp, tn, fn):
    p = 0.0
    r = 0.0
    f1 = 0.0
    a = 0.0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0

    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0

    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0

    try:
        a = (tp + tn) / (tp + fp + fn + tn)
    except ZeroDivisionError:
        a = 0

    return round(p, 3), round(r, 3), round(f1, 3), round(a, 3)
