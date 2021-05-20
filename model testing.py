import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, cross_validate
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from tpot import TPOTClassifier


def run_model(pred_field='win_rate'):
    print("Starting %s\n" % pred_field)

    dataset = 'C:/Users/bruge/PycharmProjects/RP2020/dataset_modificado/teste.csv'
    X = pd.read_csv(dataset, index_col=False)
    y = np.array(X[pred_field])
    X = X.drop(pred_field, axis=1)

    model = StackingClassifier(
        [('0', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())), ('1', MLPClassifier(learning_rate='adaptive', activation='relu')), ('2', BaggingClassifier(n_estimators=100)), ('3', GradientBoostingClassifier()), ('4', RandomForestClassifier())],
        final_estimator=LogisticRegression()
    )
    # model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    # model = MLPClassifier(learning_rate='adaptive', activation='relu')
    # model = BaggingClassifier(n_estimators=100)
    # model = GradientBoostingClassifier()
    print(np.mean(cross_val_score(model, X, y, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.3), n_jobs=-1, scoring='f1_macro')))


if __name__ == '__main__':
    problem = 'state'
    run_model(pred_field=problem)
