import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, cross_validate
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from tpot import TPOTClassifier


def select_features(X: pd.DataFrame, y: np.ndarray, n_features: int, method) -> pd.DataFrame:
    selector = SelectKBest(method, k=n_features)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    X = pd.DataFrame(X_new, columns=[f for b, f in zip(mask, X.columns) if b])
    return X


def run_model(pred_field='win_rate'):
    print("Starting %s\n" % pred_field)

    dataset = 'C:/Users/bruge/PycharmProjects/RP2020/dataset_modificado/teste.csv'
    X = pd.read_csv(dataset, index_col=False)
    y = np.array(X[pred_field])
    X = X.drop(pred_field, axis=1)

    models = [
        MLPClassifier(hidden_layer_sizes=(250,)),
        AdaBoostClassifier(MLPClassifier(hidden_layer_sizes=(250,)))
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
        # AdaBoostClassifier(),
        # BaggingClassifier(),
        # RidgeClassifier(),
        # LogisticRegression(),
        # PassiveAggressiveClassifier(),
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
        # ExtraTreeClassifier(),
        # ExtraTreesClassifier(),
        # GaussianNB,
        # GaussianProcessClassifier()
    ]
    # ('RandomForestClassifier()', 200, 200, (5162, 200), 0.7749186784282727)

    results = []
    for model in models:
        for n_feat in ['all']:
            for pca_per in [0.95, n_feat]:
                try:
                    X_new = select_features(X, y, n_feat, f_classif)
                    if pca_per != n_feat:
                        X_new = PCA(n_components=pca_per).fit_transform(X_new, y)
                    res = cross_val_score(model, X_new, y, cv=StratifiedShuffleSplit(test_size=0.3, n_splits=5), n_jobs=-1, scoring='f1_macro')
                    results.append((str(model), n_feat, pca_per, X_new.shape, np.min(res)))
                    print(results[-1])
                    print(np.mean(res), np.std(res))
                except:
                    print("Rip failed this one %s" % str(model))
    for res in sorted(results, key=lambda x: x[-1], reverse=True):
        print(res)


if __name__ == '__main__':
    problem = 'state'
    run_model(pred_field=problem)
