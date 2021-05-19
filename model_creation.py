import pandas as pd
import numpy as np
from typing import *

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from MDC import MDC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, FunctionTransformer
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import kruskal
from feature_selection import *

from seaborn import heatmap
import matplotlib.pyplot as plt
import io
import base64
import urllib

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, cross_validate, StratifiedKFold, train_test_split


def get_plot_uri():
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.close(fig=fig)
    return uri


def confmat_heatmap(conf_mat):
    if len(conf_mat) > 5:
        plt.figure(figsize=(12, 12))
    else:
        plt.figure()
    heatmap(conf_mat, annot=True, fmt="1.2f")
    return get_plot_uri()


def perclass_sens_spec(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    conf_mat = multilabel_confusion_matrix(y, y_pred)
    sens_spec = {
        i: {
            "sens": mat[0, 0] / (mat[0, 0] + mat[0, 1]) * 100,
            "spec": mat[1, 1] / (mat[1, 1] + mat[1, 0]) * 100
        } for i, mat in enumerate(conf_mat)
    }
    return sens_spec


def classifier_validation(X, y, model, validation_splitter):
    model = eval(model)
    validation_splitter = eval(validation_splitter)
    scores = cross_validate(model, X, y, cv=validation_splitter, scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], n_jobs=-1)
    scores = {k: (np.mean(v) * 100, np.std(v) * 100) for k, v in scores.items()}
    sens_spec = perclass_sens_spec(model, X, y)
    y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    conf_mat = confusion_matrix(y, y_pred, normalize='true')
    return scores, sens_spec, conf_mat
