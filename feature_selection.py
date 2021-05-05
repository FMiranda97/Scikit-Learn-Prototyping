import base64
import io
import urllib
from typing import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import kruskal

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import CSS4_COLORS
import matplotlib.lines as mlines

import seaborn
from mpl_toolkits.mplot3d import Axes3D  # unused import, but necessary to activate some 3d plot thingy


def plot_correlation(X: pd.DataFrame):
    seaborn.heatmap(X.corr(), cmap="cool")
    plt.tight_layout()
    plt.show()


def get_plot_uri():
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.close(fig=fig)
    return uri


def plot_features(X: pd.DataFrame, y: np.ndarray, interactive: bool = True, n_feat=None) -> str:
    n_samples = X.shape[0]
    if n_feat is None:
        n_features = X.shape[1]
    else:
        n_features = n_feat
    n_classes = len(np.unique(y))
    if n_classes == 2:
        colors = matplotlib.colors.ListedColormap(['red', 'green'])
    elif n_classes == 3:
        colors = matplotlib.colors.ListedColormap(['red', 'green', 'blue'])
    else:
        colors = plt.get_cmap("tab20")
    if n_samples < 1 or n_features < 1 or n_features > 4:
        return ''
    if n_features == 1:
        fig = plt.figure()
        plt.title("Distribution of Classes According to the Best Feature")
        plt.xlabel(X.columns[0])
        fig.get_axes()[0].get_yaxis().set_visible(False)
        plt.scatter(X[X.columns[0]], [0 for _ in range(n_samples)], c=y, cmap=colors)
        handles = [mlines.Line2D([], [], color=colors(i), marker='o', linestyle='None', markersize=10, label='Class %d' % i) for i in range(n_classes)]
        plt.legend(handles=handles, bbox_to_anchor=(-0.05, 1))
        plt.tight_layout()
        return get_plot_uri()
    elif n_features == 2:
        fig = plt.figure()
        plt.title("Distribution of Classes According to the 2 Best Features")
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap=colors)
        handles = [mlines.Line2D([], [], color=colors(i), marker='o', linestyle='None', markersize=10, label='Class %d' % i) for i in range(n_classes)]
        plt.legend(handles=handles, bbox_to_anchor=(-0.05, 1))
        plt.tight_layout()
        return get_plot_uri()
    elif n_features == 3:
        if interactive:
            matplotlib.use('Qt4Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Distribution of Classes According to the 3 Best Features")
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel(X.columns[2])
        ax.scatter(X[X.columns[0]], X[X.columns[1]], X[X.columns[2]], c=y, cmap=colors)
        handles = [mlines.Line2D([], [], color=colors(i), marker='o', linestyle='None', markersize=10, label='Class %d' % i) for i in range(n_classes)]
        plt.legend(handles=handles, bbox_to_anchor=(-0.05, 1))
        # plt.tight_layout()
        return get_plot_uri()
    elif n_features == 4:
        if interactive:
            matplotlib.use('Qt4Agg')
        markers = ["P", "o", "1", "s", "p", "+", "*", "x", "D", "2", "v", "3", "^", "4", "<", "d", ">", "X", "."]
        markers = markers[0:n_classes]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, m in enumerate(markers):
            mask = y == i
            # mask is now an array of booleans that can be used for indexing
            img = ax.scatter(X[mask][X.columns[0]], X[mask][X.columns[1]], X[mask][X.columns[2]], c=X[mask][X.columns[3]], marker=m, cmap=plt.cool())

        handles = [mlines.Line2D([], [], color='black', marker=m, linestyle='None', markersize=10, label='Class %d' % i) for i, m in enumerate(markers)]
        plt.legend(handles=handles, bbox_to_anchor=(-0.05, 1))
        # plt.tight_layout()
        cax = fig.add_axes([0.15, .95, 0.7, 0.03])
        cb = fig.colorbar(img, orientation='horizontal', cax=cax)
        cb.set_label(X.columns[3])
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel(X.columns[2])
        return get_plot_uri()


def filter_correlated(X: pd.DataFrame, threshold: float = 0.5, limit=None) -> pd.DataFrame:
    X_new = X
    while True:
        if limit is not None and X_new.shape[1] <= limit:
            break
        cor = abs(np.corrcoef(X_new.transpose()))
        correlation_values = {i: [cor[i][j] for j in range(cor.shape[0]) if i != j] for i in range(cor.shape[0])}
        correlated_count = {k: len([i for i in v if i > threshold]) for k, v in correlation_values.items()}
        if sum(correlated_count.values()) == 0:
            break
        most_correlations = max(correlated_count.values())
        candidates = [i for i, count in correlated_count.items() if count == most_correlations]
        candidates_total_correlation = {i: np.nansum(correlation_values[i]) for i in candidates}
        maximum = max(candidates_total_correlation.values())
        candidate_to_remove = random.choice([k for k, v in candidates_total_correlation.items() if v == maximum])
        try:
            X_new = X_new.drop(X_new.columns[candidate_to_remove], axis=1)
        except:
            X_new = X_new[:, [False if x == candidate_to_remove else True for x in range(X_new.shape[1])]]
    return X_new


def fisher_score(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    unique_classes = np.unique(y)
    n_feat = X.shape[1]
    n_samples = X.shape[0]
    scores = []
    for feature in range(n_feat):
        m = np.mean(X[:, feature])
        classes = [X[y == i, feature] for i in unique_classes]
        top = sum([(len(c) / n_samples) * (np.mean(c) - m) ** 2 for c in classes])
        bottom = sum([(len(c) / n_samples) * np.var(c) for c in classes])
        scores.append(top / bottom)
    return np.array(scores)


def ks_classif(X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_classes = np.unique(y)
    n_feat = X.shape[1]
    scores = []
    pvalues = []
    for feature in range(n_feat):
        args = [X[y == i, feature] for i in unique_classes]
        score, pvalue = kruskal(*args)
        scores.append(score)
        pvalues.append(pvalue)
    return np.array(scores), np.array(pvalues)


def ks_classif_correlation_penalty(X: pd.DataFrame, y: np.ndarray, penalty_coefficient: float = 1.73) -> Tuple[np.ndarray, np.ndarray]:
    scores, pvalues = ks_classif(X, y)
    scores = correlation_penalty(X, scores, penalty_coefficient=penalty_coefficient)
    return np.array(scores), np.array(pvalues)


def correlation_penalty(X: pd.DataFrame, scores: [np.ndarray, list], penalty_coefficient: float = 1.73) -> list:
    n_feat = X.shape[1]
    corr_matrix = abs(np.corrcoef(X.transpose()))
    already_penalized = []
    for i in range(n_feat):
        sorted_indexes = np.argsort(-1 * np.array(scores))
        already_penalized.append(sorted_indexes[i])
        for j in range(n_feat):
            if sorted_indexes[i] != j and j not in already_penalized:
                scores[j] = (1 - corr_matrix[sorted_indexes[i], j] ** penalty_coefficient) * scores[j]
    return scores


def auc_classif(X: pd.DataFrame, y: np.ndarray, aggregate_func=sum) -> np.ndarray:
    n_classes = len(np.unique(y))
    n_feat = X.shape[1]
    scores = []
    y = label_binarize(y, classes=np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    for feat in range(n_feat):
        clf = OneVsRestClassifier(LinearSVC(random_state=0))
        y_score = clf.fit(X_train[:, feat].reshape(-1, 1), y_train).decision_function(X_test[:, feat].reshape(-1, 1))
        roc_auc = []
        if n_classes <= 2:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc.append(auc(fpr, tpr))
        else:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc.append(auc(fpr, tpr))
        scores.append(aggregate_func(roc_auc))
    return np.array(scores)


def auc_sum_classif(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    return auc_classif(X, y)


def auc_mul_classif(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    return auc_classif(X, y, aggregate_func=np.prod)


def auc_sum_classif_correlation_penalty(X: pd.DataFrame, y: np.ndarray, penalty_coefficient: float = 1.73) -> np.ndarray:
    scores = auc_classif(X, y)
    scores = correlation_penalty(X, scores, penalty_coefficient=penalty_coefficient)
    return np.array(scores)


def auc_mul_classif_correlation_penalty(X: pd.DataFrame, y: np.ndarray, penalty_coefficient: float = 1.73) -> np.ndarray:
    scores = auc_classif(X, y, aggregate_func=np.prod)
    scores = correlation_penalty(X, scores, penalty_coefficient=penalty_coefficient)
    return np.array(scores)


def normalize(scores: np.ndarray) -> np.ndarray:
    max_value = scores.max()
    min_value = scores.min()
    scores = (scores - min_value) / (max_value - min_value)
    return scores


def multi_stats_classif(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    scores_auc = normalize(auc_classif(X, y, aggregate_func=np.prod))
    scores_ks = normalize(ks_classif(X, y)[0])
    scores_chi2 = normalize(chi2(X, y)[0])
    return normalize(np.array([x * y * z for x, y, z in zip(scores_auc, scores_ks, scores_chi2)]))


def multi_stats_classif_correlation_penalty(X: pd.DataFrame, y: np.ndarray, penalty_coefficient: float = 2) -> np.ndarray:
    scores = multi_stats_classif(X, y)
    scores = correlation_penalty(X, scores, penalty_coefficient=penalty_coefficient)
    return np.array(scores)


def feature_filtering(X: pd.DataFrame, y: np.ndarray, n_feat: int = 3, method: int = 1) -> Tuple[
    pd.DataFrame, np.ndarray, dict]:
    n_feat = min(n_feat, len(X.columns))
    if method > 0 and method < 13:
        selector = eval(get_feature_filtering_model(method, n_feat))
    else:
        return X, y, {}
    selector = selector.fit(X, y)
    scores = {X.columns[i]: score for i, score in enumerate(selector.scores_)}
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    X_new = selector.transform(X)
    mask = selector.get_support()
    X_new = pd.DataFrame(X_new, columns=[f for b, f in zip(mask, X.columns) if b])
    order = [k for i, k in enumerate(scores.keys()) if i < X_new.shape[1]]
    X_new = X_new.reindex(columns=order)
    return X_new, y, scores


def get_feature_filtering_model(method, n_feat):
    if method == 1:
        selector = 'SelectKBest(chi2, k=%d)' % n_feat
    elif method == 2:
        selector = 'SelectKBest(f_classif, k=%d)' % n_feat
    elif method == 3:
        selector = 'SelectKBest(ks_classif, k=%d)' % n_feat
    elif method == 4:
        selector = 'SelectKBest(mutual_info_classif, k=%d)' % n_feat
    elif method == 5:
        selector = 'SelectKBest(ks_classif_correlation_penalty, k=%d)' % n_feat
    elif method == 6:
        selector = 'SelectKBest(auc_sum_classif, k=%d)' % n_feat
    elif method == 7:
        selector = 'SelectKBest(auc_mul_classif, k=%d)' % n_feat
    elif method == 8:
        selector = 'SelectKBest(auc_sum_classif_correlation_penalty, k=%d)' % n_feat
    elif method == 9:
        selector = 'SelectKBest(auc_mul_classif_correlation_penalty, k=%d)' % n_feat
    elif method == 10:
        selector = 'SelectKBest(multi_stats_classif, k=%d)' % n_feat
    elif method == 11:
        selector = 'SelectKBest(multi_stats_classif_correlation_penalty, k=%d)' % n_feat
    elif method == 12:
        selector = 'SelectKBest(fisher_score, k=%d)' % n_feat
    else:
        selector = None
    assert selector is not None
    return selector


def get_method_name(method: int) -> str:
    if method == 1:
        return "Chi-Squared"
    elif method == 2:
        return "F-statistic"
    elif method == 3:
        return "Kolmogorov–Smirnov Test"
    elif method == 4:
        return "Mutual Information"
    elif method == 5:
        return "Kolmogorov–Smirnov Test with Correlation Penalty"
    elif method == 6:
        return "Sum of Areas Under ROC curve"
    elif method == 7:
        return "Multiplication of Areas Under ROC curve"
    elif method == 8:
        return "Sum of Areas Under ROC curve with Correlation Penalty"
    elif method == 9:
        return "Multiplication of Areas Under ROC curve with Correlation Penalty"
    elif method == 10:
        return "AUC-KS-Chi2 normalized multiplication"
    elif method == 11:
        return "AUC-KS-Chi2 normalized multiplication with Correlation Penalty"
    elif method == 12:
        return "Fisher Score"
    elif method == 13:
        return "Principal Component Analysis"
    elif method == 14:
        return "Linear Discrimant Analysis"


def feature_reduction(X: pd.DataFrame, y: np.ndarray, n_feat: int = 3, method: int = 1) -> Tuple[
    pd.DataFrame, np.ndarray]:
    if method == 1:
        X_new = PCA(n_components=n_feat).fit_transform(X)
    elif method == 2:
        X_new = LinearDiscriminantAnalysis(n_components=n_feat).fit_transform(X, y)
    else:
        return X, y
    X_new = pd.DataFrame(X_new, columns=["Feature %d" % (i + 1) for i in range(X_new.shape[1])])
    return X_new, y


def get_feature_reduction_model(method, n_feat):
    if method == 1:
        return PCA(n_components=n_feat)
    elif method == 2:
        return LinearDiscriminantAnalysis(n_components=n_feat)
    else:
        return None


def feature_analysis(X: pd.DataFrame, y: np.ndarray):
    classes = np.unique(y)
    features = X.columns
    results = {
        f: {
            c: {
                'mean': X[y == c][f].mean(),
                'std': X[y == c][f].std()
            } for c in classes
        } for f in features
    }
    return results
