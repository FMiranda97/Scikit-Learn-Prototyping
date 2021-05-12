from io import StringIO

import pandas as pd
import numpy as np
import sqlite3

from sklearn.impute import SimpleImputer


def get_data(data_set) -> (pd.DataFrame, np.ndarray):
    # TODO get latest loaded csv
    df = pd.read_csv(StringIO(data_set))
    y = np.array(df['state'])
    X = df.drop(['state'], axis=1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[X.columns] = imp.fit_transform(X, y)
    return X, y

def get_data_from_csv(data_set) -> (pd.DataFrame, np.ndarray):
    # TODO get latest loaded csv
    df = pd.read_csv(data_set, index_col=False)
    y = np.array(df['state'])
    X = df.drop(['state'], axis=1)
    return X, y


# if __name__ == '__main__':
#     get_data()
