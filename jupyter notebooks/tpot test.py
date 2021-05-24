import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


def run_tpot(pred_field='win_rate', max_time_mins=1200):
    print("Starting %s\n" % pred_field)

    dataset = 'C:/Users/bruge/PycharmProjects/RP2020/dataset_modificado/teste.csv'
    X = pd.read_csv(dataset, index_col=False)
    y = np.array(X[pred_field])

    X = X.drop(pred_field, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    tpot = TPOTClassifier(config_dict='TPOT NN', verbosity=3, n_jobs=-1, max_time_mins=max_time_mins, generations=None, warm_start=True, scoring='f1_macro', max_eval_time_mins=2)
    tpot.fit(X_train, y_train)

    # print part of pipeline dictionary

    print("f1_macro = %s\n" % tpot.score(X_test, y_test))
    print("\n\n%s\n\n" % str(tpot))

    print("\n\n%s\n\n" % tpot.export())
    tpot.export('tpot_classifier_%s.py' % pred_field)


if __name__ == '__main__':
    problem = 'state'
    run_tpot(pred_field=problem)

# -1	0.7845902803904253	RandomForestClassifier(input_matrix, RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=gini, RandomForestClassifier__max_features=0.45, RandomForestClassifier__min_samples_leaf=1, RandomForestClassifier__min_samples_split=5, RandomForestClassifier__n_estimators=100)
#
# -2	0.8088411253025793	GradientBoostingClassifier(RandomForestClassifier(input_matrix, RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=gini, RandomForestClassifier__max_features=0.25, RandomForestClassifier__min_samples_leaf=4, RandomForestClassifier__min_samples_split=8, RandomForestClassifier__n_estimators=100), GradientBoostingClassifier__learning_rate=0.1, GradientBoostingClassifier__max_depth=2, GradientBoostingClassifier__max_features=0.2, GradientBoostingClassifier__min_samples_leaf=10, GradientBoostingClassifier__min_samples_split=10, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.3)
#
# -3	0.8160093077100796	GradientBoostingClassifier(MinMaxScaler(RandomForestClassifier(input_matrix, RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=gini, RandomForestClassifier__max_features=0.25, RandomForestClassifier__min_samples_leaf=4, RandomForestClassifier__min_samples_split=7, RandomForestClassifier__n_estimators=100)), GradientBoostingClassifier__learning_rate=0.1, GradientBoostingClassifier__max_depth=1, GradientBoostingClassifier__max_features=0.9500000000000001, GradientBoostingClassifier__min_samples_leaf=18, GradientBoostingClassifier__min_samples_split=5, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.4)
#
# -7	0.8166218345053897	GradientBoostingClassifier(SGDClassifier(RandomForestClassifier(CombineDFs(input_matrix, RandomForestClassifier(Binarizer(PCA(GaussianNB(input_matrix), PCA__iterated_power=4, PCA__svd_solver=randomized), Binarizer__threshold=1.0), RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=entropy, RandomForestClassifier__max_features=1.0, RandomForestClassifier__min_samples_leaf=4, RandomForestClassifier__min_samples_split=11, RandomForestClassifier__n_estimators=100)), RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=gini, RandomForestClassifier__max_features=0.25, RandomForestClassifier__min_samples_leaf=4, RandomForestClassifier__min_samples_split=5, RandomForestClassifier__n_estimators=100), SGDClassifier__alpha=0.01, SGDClassifier__eta0=1.0, SGDClassifier__fit_intercept=True, SGDClassifier__l1_ratio=0.5, SGDClassifier__learning_rate=invscaling, SGDClassifier__loss=hinge, SGDClassifier__penalty=elasticnet, SGDClassifier__power_t=0.1), GradientBoostingClassifier__learning_rate=0.1, GradientBoostingClassifier__max_depth=1, GradientBoostingClassifier__max_features=0.15000000000000002, GradientBoostingClassifier__min_samples_leaf=18, GradientBoostingClassifier__min_samples_split=5, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.4)
