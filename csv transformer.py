import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

if __name__ == '__main__':
    input_filename = 'dataset_modificado/PatientInfo_modificado.csv'
    output_filename = 'dataset_modificado/teste.csv'

    knn_weights = 'uniform'  # weights{‘uniform’, ‘distance’} or callable, default=’uniform’
    knn_neighbors = 5
    categorical_to_many_columns = True
    cenario = 'C'

    features_to_exclude = []
    # features_to_exclude = ['sex', 'age', 'country', 'province', 'city', 'infection_case', 'confirmed_date', 'state', 'latitude', 'longitude', 'elementary_school_count', 'kindergarten_count', 'university_count', 'academy_ratio', 'elderly_population_ratio', 'elderly_alone_ratio', 'nursing_home_count', 'search_cold', 'search_flu', 'search_pneumonia', 'search_coronavirus', 'confirmed_cases', 'deceased_cases', 'confirmed_cases_male', 'deceased_cases_male', 'confirmed_cases_female', 'deceased_cases_female', 'deceased_age_0s', 'confirmed_age_0s', 'deceased_age_10s', 'confirmed_age_10s', 'deceased_age_20s', 'confirmed_age_20s', 'deceased_age_30s', 'confirmed_age_30s', 'deceased_age_40s', 'confirmed_age_40s', 'deceased_age_50s', 'confirmed_age_50s', 'deceased_age_60s', 'confirmed_age_60s', 'deceased_age_70s', 'confirmed_age_70s', 'deceased_age_80s', 'confirmed_age_80s', 'confirmed_Seoul', 'released_Seoul', 'deceased_Seoul', 'confirmed_Busan', 'released_Busan', 'deceased_Busan', 'confirmed_Daegu', 'released_Daegu', 'deceased_Daegu', 'confirmed_Incheon', 'released_Incheon', 'deceased_Incheon', 'confirmed_Gwangju', 'released_Gwangju', 'deceased_Gwangju', 'confirmed_Daejeon', 'released_Daejeon', 'deceased_Daejeon', 'confirmed_Ulsan', 'released_Ulsan', 'deceased_Ulsan', 'confirmed_Sejong', 'released_Sejong', 'deceased_Sejong', 'confirmed_Gyeonggi_do', 'released_Gyeonggi_do', 'deceased_Gyeonggi_do', 'confirmed_Gangwon_do', 'released_Gangwon_do', 'deceased_Gangwon_do', 'confirmed_Chungcheongbuk_do', 'released_Chungcheongbuk_do', 'deceased_Chungcheongbuk_do', 'confirmed_Chungcheongnam_do', 'released_Chungcheongnam_do', 'deceased_Chungcheongnam_do', 'confirmed_Jeollabuk_do', 'released_Jeollabuk_do', 'deceased_Jeollabuk_do', 'confirmed_Jeollanam_do', 'released_Jeollanam_do', 'deceased_Jeollanam_do', 'confirmed_Gyeongsangbuk_do', 'released_Gyeongsangbuk_do', 'deceased_Gyeongsangbuk_do', 'confirmed_Gyeongsangnam_do', 'released_Gyeongsangnam_do', 'deceased_Gyeongsangnam_do', 'confirmed_Jeju_do', 'released_Jeju_do', 'deceased_Jeju_do', 'tests_count', 'tests_negative', 'tests_positive', 'tests_confirmed', 'tests_released', 'tests_deceased', 'avg_temp', 'min_temp', 'max_temp', 'max_wind_speed', 'most_wind_direction', 'avg_relative_humidity']

    df = pd.read_csv(input_filename, index_col=False)

    features = df.columns.tolist()
    for f in features_to_exclude:
        features.remove(f)
    df = df[features]

    # encode categorical
    categorical_cols = df.columns[df.dtypes == object].tolist()
    categorical_cols.remove('state')
    if categorical_to_many_columns:
        df = pd.concat([df.drop(categorical_cols, axis=1), pd.get_dummies(df[categorical_cols], prefix=categorical_cols)], axis=1)
    else:
        for category in categorical_cols:
            df[category] = LabelEncoder().fit_transform(df[category])

    # scale and input missing values
    feature_cols = df.columns.to_list()
    feature_cols.remove('state')
    df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])
    df[feature_cols] = KNNImputer(n_neighbors=knn_neighbors, weights=knn_weights).fit_transform(df[feature_cols])

    # transform class column
    if cenario == 'A':
        df['state'] = [1 if x == 'released' else 0 for x in df['state']]
    elif cenario == 'B':
        df['state'] = [1 if x == 'deceased' else 0 for x in df['state']]
    elif cenario == 'C':
        df['state'] = [['released', 'isolated', 'deceased'].index(x) for x in df['state']]
    df = df.sample(frac=1)
    df.to_csv(output_filename, index=False)
