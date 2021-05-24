import pandas as pd
import os
import numpy as np

def modificar_ficheiro(df):
    for feature_name in df.columns:
        if feature_name == 'sex' or feature_name == 'sex_desvio_padrao' or feature_name == 'sex_mediana' \
                or feature_name == 'sex_moda' or feature_name == 'sex_mil' or feature_name == 'sex_zero':
            df[feature_name] = df[feature_name].replace({'male': 0, 'female': 1})
            df[feature_name] = pd.to_numeric(df[feature_name])

        if feature_name == 'age' or feature_name == 'age_desvio_padrao' or feature_name == 'age_moda' \
            or feature_name == 'age_mediana' or feature_name == 'age_mil' or feature_name == 'age_zero':
            df[feature_name]=df[feature_name].str.split("s", n=0, expand=True)
            df[feature_name]=pd.to_numeric(df[feature_name])

        if feature_name == 'latitude' or feature_name =='longitude' or feature_name =='elementary_school_count' or \
            feature_name =='kindergarten_count' or feature_name =='university_count' or feature_name =='academy_ratio' \
            or feature_name == 'elderly_population_ratio' or feature_name == 'elderly_alone_ratio' or \
            feature_name == 'nursing_home_count' or feature_name == 'search_cold' or feature_name == 'search_flu' \
            or feature_name =='search_pneumonia' or feature_name == 'search_coronavirus' or feature_name == \
            'confirmed_cases' or feature_name == 'deceased_cases' or feature_name =='confirmed_cases_male' or \
            feature_name == 'deceased_cases_male' or feature_name == 'confirmed_cases_female' or feature_name == \
            'deceased_cases_female' or feature_name == 'deceased_age_0s' or feature_name == 'confirmed_age_0s' or \
            feature_name == 'deceased_age_10s' or feature_name == 'confirmed_age_10s' or feature_name == \
            'deceased_age_20s' or feature_name == 'confirmed_age_20s' or feature_name ==  'deceased_age_30s' or \
            feature_name == 'confirmed_age_30s' or feature_name ==  'deceased_age_40s' or feature_name == \
            'confirmed_age_40s' or feature_name == 'deceased_age_50s' or feature_name == 'confirmed_age_50s' or \
            feature_name =='deceased_age_60s' or feature_name == 'confirmed_age_60s' or feature_name == \
            'deceased_age_70s' or feature_name == 'confirmed_age_70s' or feature_name == 'deceased_age_80s' or \
            feature_name == 'confirmed_age_80s' or feature_name ==  'confirmed_Seoul' or feature_name ==  \
            'released_Seoul' or feature_name ==  'deceased_Seoul' or feature_name == 'confirmed_Busan' or \
            feature_name ==  'released_Busan' or feature_name == 'deceased_Busan' or feature_name == \
            'confirmed_Daegu' or feature_name ==  'released_Daegu' or feature_name == 'deceased_Daegu' or \
            feature_name ==  'confirmed_Incheon' or feature_name ==  'released_Incheon' or feature_name == \
            'deceased_Incheon' or feature_name == 'confirmed_Gwangju' or feature_name == 'released_Gwangju' or \
            feature_name == 'deceased_Gwangju' or feature_name == 'confirmed_Daejeon' or feature_name == \
            'released_Daejeon' or feature_name == 'deceased_Daejeon' or feature_name == 'confirmed_Ulsan' or \
            feature_name ==  'released_Ulsan' or feature_name == 'deceased_Ulsan' or feature_name == \
            'confirmed_Sejong' or feature_name == 'released_Sejong' or feature_name ==  'deceased_Sejong' or \
            feature_name ==  'confirmed_Gyeonggi_do' or feature_name ==  'released_Gyeonggi_do' or feature_name == \
            'deceased_Gyeonggi_do' or feature_name ==  'confirmed_Gangwon_do' or feature_name == \
            'released_Gangwon_do' or feature_name ==  'deceased_Gangwon_do' or feature_name == \
            'confirmed_Chungcheongbuk_do' or feature_name == 'released_Chungcheongbuk_do' or feature_name ==\
            'deceased_Chungcheongbuk_do' or feature_name ==  'confirmed_Chungcheongnam_do' or feature_name ==  \
            'released_Chungcheongnam_do' or feature_name ==  'deceased_Chungcheongnam_do' or \
            feature_name == 'confirmed_Jeollabuk_do' or feature_name =='released_Jeollabuk_do' or \
            feature_name ==  'deceased_Jeollabuk_do' or feature_name ==  'confirmed_Jeollanam_do' or \
            feature_name ==  'released_Jeollanam_do' or feature_name ==  'deceased_Jeollanam_do' or \
            feature_name ==  'confirmed_Gyeongsangbuk_do' or feature_name ==  'released_Gyeongsangbuk_do' or \
            feature_name ==  'deceased_Gyeongsangbuk_do' or feature_name ==  'confirmed_Gyeongsangnam_do' or \
            feature_name ==  'released_Gyeongsangnam_do' or feature_name ==  'deceased_Gyeongsangnam_do' or \
            feature_name ==  'confirmed_Jeju_do' or feature_name == 'released_Jeju_do' or \
            feature_name ==  'deceased_Jeju_do' or feature_name ==   'tests_count' or feature_name == 'tests_negative' \
            or feature_name ==  'tests_positive' or feature_name ==   'tests_confirmed' or feature_name == \
            'tests_released' or feature_name ==  'tests_deceased' or feature_name == 'avg_temp' or \
            feature_name == 'min_temp' or feature_name == 'max_temp' or feature_name == 'max_wind_speed' or \
            feature_name == 'most_wind_direction' or feature_name == 'avg_relative_humidity':

            df[feature_name]=pd.to_numeric(df[feature_name])


        if feature_name == 'confirmed_date' or feature_name == 'start_date' or feature_name == 'end_date' \
            or feature_name == 'date':
            data = df[feature_name].str.split("-", n=0, expand=True)
            df[feature_name] = pd.to_numeric(data[0] + data[1] + data[2])

        # if feature_name == 'country' or feature_name == 'type' or feature_name == 'gov_policy' \
        #         or feature_name == 'detail' or feature_name == 'province' or feature_name == 'city' \
        #         or feature_name == 'infection_case' or feature_name == 'city_desvio_padrao' \
        #         or feature_name == 'city_mediana' or feature_name == 'state' or feature_name == 'city_moda' or feature_name == 'city_mil' \
        #         or feature_name == 'city_zero' or feature_name == 'infection_case_desvio_padrao' \
        #         or feature_name == 'infection_case_mediana' or feature_name == 'infection_case_moda' \
        #         or feature_name == 'infection_case_zero' or feature_name == 'infection_case_mil':
        #     features = df[feature_name].dropna().unique() #ignorar os nans
        #     cont = 0
        #     for lista in features:
        #         df[feature_name] = df[feature_name].replace({lista: cont})
        #         cont = cont + 1

        if feature_name == 'group':
            df[feature_name] = df[feature_name].replace({True: 1, False: 0})

    return df

def medias_weather():
    df = pd.read_csv(os.getcwd() + "\\dataset\\Weather.csv")

    # média da coluna avg_temp em relação á province
    # ou seja, faz-se a média de todos os valores "avg_temp" em cada province
    mean_avg_time = df.groupby(['province', 'date'])["avg_temp"].mean()
    mean_min_temp = df.groupby(['province', 'date'])["min_temp"].mean()
    mean_max_temp = df.groupby(['province', 'date'])["max_temp"].mean()
    mean_max_wind_speed = df.groupby(['province', 'date'])["max_wind_speed"].mean()
    mean_most_wind_direction = df.groupby(['province', 'date'])["most_wind_direction"].mean()
    mean_avg_relative_humidity = df.groupby(['province', 'date'])["avg_relative_humidity"].mean()

    table = pd.concat(
        [mean_min_temp, mean_avg_time, mean_max_temp, mean_max_wind_speed, mean_most_wind_direction,
         mean_avg_relative_humidity], axis=1)

    table.to_csv(os.getcwd() + "\\dataset_modificado\\medias_Weather.csv")


def saber_media(nome_coluna, df):
    media = df[nome_coluna].mean()
    return media

def saber_moda(nome_coluna, df):
    moda = df[nome_coluna].mode()
    return moda[0]

def saber_mediana(nome_coluna, df):
    mediana = df[nome_coluna].median()
    return mediana

def saber_desvio(nome_coluna, df):
    desvio = df[nome_coluna].std()
    return desvio

def definir_colunas_valor(nome_coluna, df, valor):
    df[nome_coluna].fillna(valor, inplace=True)
    return df

def delete_column(nome_coluna, df):
    df = df.drop(nome_coluna, axis=1)
    return df

def delete_rows(numero_linha, df):
    df = df.drop(numero_linha, axis=0)
    return df

def add_column(nome_coluna, df, metodo):
    df.insert(df.columns.get_loc(nome_coluna) + 1, nome_coluna + metodo, df[nome_coluna])
    return df

def verify_nan_values(df):
    for i in range(len(df)):
        if df['city'] is np.nan:
            print(i)

def missing_values(nome_ficheiro):
    df = pd.read_csv(os.getcwd() + nome_ficheiro)

    df = delete_column('infected_by', df)
    df = delete_column('contact_number', df)
    df = delete_column('deceased_date', df)
    df = delete_column('released_date', df)
    df = delete_column('symptom_onset_date', df)
    df = delete_column('patient_id', df)

    df = add_column('sex', df, '_moda')
    df = add_column('sex', df, '_mediana')
    df = add_column('sex', df, '_desvio_padrao')
    df = add_column('sex', df, '_zero')
    df = add_column('sex', df, '_mil')
    df = add_column('age', df, '_moda')
    df = add_column('age', df, '_mediana')
    df = add_column('age', df, '_desvio_padrao')
    df = add_column('age', df, '_zero')
    df = add_column('age', df, '_mil')
    df = add_column('city', df, '_moda')
    df = add_column('city', df, '_mediana')
    df = add_column('city', df, '_desvio_padrao')
    df = add_column('city', df, '_zero')
    df = add_column('city', df, '_mil')
    df = add_column('infection_case', df, '_moda')
    df = add_column('infection_case', df, '_mediana')
    df = add_column('infection_case', df, '_desvio_padrao')
    df = add_column('infection_case', df, '_zero')
    df = add_column('infection_case', df, '_mil')

    df = delete_rows(4730, df)
    df = delete_rows(4731, df)
    df = delete_rows(4732, df)
    df.reset_index(drop=True, inplace=True)

    df = modificar_ficheiro(df)

    df = definir_colunas_valor('age_zero', df, 0)
    df = definir_colunas_valor('age_mil', df, -1000)
    df = definir_colunas_valor('age_moda', df, saber_moda('age', df))
    df = definir_colunas_valor('age_mediana', df, saber_mediana('age', df))
    df = definir_colunas_valor('age_desvio_padrao', df, saber_desvio('age', df))
    df = definir_colunas_valor('age', df, saber_media('age', df))

    df = definir_colunas_valor('sex_zero', df, 0)
    df = definir_colunas_valor('sex_mil', df, -1000)
    df = definir_colunas_valor('sex_moda', df, saber_moda('sex', df))
    df = definir_colunas_valor('sex_mediana', df, saber_mediana('sex', df))
    df = definir_colunas_valor('sex_desvio_padrao', df, saber_desvio('sex', df))
    df = definir_colunas_valor('sex', df, saber_media('sex', df))

    df = definir_colunas_valor('city_zero', df, 0)
    df = definir_colunas_valor('city_mil', df, -1000)
    df = definir_colunas_valor('city_moda', df, saber_moda('city', df))
    df = definir_colunas_valor('city_mediana', df, saber_mediana('city', df))
    df = definir_colunas_valor('city_desvio_padrao', df, saber_desvio('city', df))
    df = definir_colunas_valor('city', df, saber_media('city', df))

    df = definir_colunas_valor('infection_case_zero', df, 0)
    df = definir_colunas_valor('infection_case_mil', df, -1000)
    df = definir_colunas_valor('infection_case_moda', df, saber_moda('infection_case', df))
    df = definir_colunas_valor('infection_case_mediana', df, saber_mediana('infection_case', df))
    df = definir_colunas_valor('infection_case_desvio_padrao', df, saber_desvio('infection_case', df))
    df = definir_colunas_valor('infection_case', df, saber_media('infection_case', df))

    df.to_csv(os.getcwd() + "\\dataset_modificado\\PatientInfo_Final.csv", index=False)

    return df

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def data_extract(cenario):
    if cenario.upper() == "A":
        ficheiro = pd.read_csv(os.getcwd() + "\\dataset_modificado\\PatientInfo_Final.csv", index_col=False)
        y = np.array(ficheiro['state'].replace({2: 1}))
        X = ficheiro.drop(["state"], axis=1)
        X = normalize(X)
    elif cenario.upper() == "B":
        pass
    else:
        pass
    return X, y

def data_selection():
    df_patInfo = pd.read_csv(os.getcwd() + "\\dataset\\PatientInfo.csv")
    df_medias_weather=pd.read_csv(os.getcwd() + "\\dataset_modificado\\medias_weather.csv")
    df_region=pd.read_csv(os.getcwd() + "\\dataset\\Region.csv")
    df_trend=pd.read_csv(os.getcwd() + "\\dataset\\SearchTrend.csv")
    df_tgender=pd.read_csv(os.getcwd() + "\\dataset\\TimeGender.csv")
    df_tage=pd.read_csv(os.getcwd() + "\\dataset\\TimeAge.csv")
    df_tprovince=pd.read_csv(os.getcwd() + "\\dataset\\TimeProvince.csv")
    df_time=pd.read_csv(os.getcwd() + "\\dataset\\Time.csv")

    serie_latitude=pd.Series([], dtype='object')
    serie_longitude=pd.Series([], dtype='object')
    serie_elementary=pd.Series([], dtype='object')
    serie_kindergarten=pd.Series([], dtype='object')
    serie_university=pd.Series([], dtype='object')
    serie_academy=pd.Series([], dtype='object')
    serie_elderly=pd.Series([], dtype='object')
    serie_elderly_alone=pd.Series([], dtype='object')
    serie_nursing=pd.Series([], dtype='object')

    serie_cold=pd.Series([], dtype='object')
    serie_flu=pd.Series([], dtype='object')
    serie_pneumonia=pd.Series([], dtype='object')
    serie_coronavirus=pd.Series([], dtype='object')

    serie_confirmed=pd.Series([], dtype='object')
    serie_deceased=pd.Series([], dtype='object')
    serie_confirmed_male=pd.Series([], dtype='object')
    serie_deceased_male=pd.Series([], dtype='object')
    serie_confirmed_female=pd.Series([], dtype='object')
    serie_deceased_female=pd.Series([], dtype='object')

    serie_deceased_0=pd.Series([], dtype='object')
    serie_confirmed_0=pd.Series([], dtype='object')
    serie_deceased_10=pd.Series([], dtype='object')
    serie_confirmed_10=pd.Series([], dtype='object')
    serie_deceased_20=pd.Series([], dtype='object')
    serie_confirmed_20=pd.Series([], dtype='object')
    serie_deceased_30=pd.Series([], dtype='object')
    serie_confirmed_30=pd.Series([], dtype='object')
    serie_deceased_40=pd.Series([], dtype='object')
    serie_confirmed_40=pd.Series([], dtype='object')
    serie_deceased_50=pd.Series([], dtype='object')
    serie_confirmed_50=pd.Series([], dtype='object')
    serie_deceased_60=pd.Series([], dtype='object')
    serie_confirmed_60=pd.Series([], dtype='object')
    serie_deceased_70=pd.Series([], dtype='object')
    serie_confirmed_70=pd.Series([], dtype='object')
    serie_deceased_80=pd.Series([], dtype='object')
    serie_confirmed_80=pd.Series([], dtype='object')

    serie_confirmed_Seoul=pd.Series([], dtype='object')
    serie_released_Seoul=pd.Series([], dtype='object')
    serie_deceased_Seoul=pd.Series([], dtype='object')
    serie_confirmed_Busan=pd.Series([], dtype='object')
    serie_released_Busan=pd.Series([], dtype='object')
    serie_deceased_Busan=pd.Series([], dtype='object')
    serie_confirmed_Daegu=pd.Series([], dtype='object')
    serie_released_Daegu=pd.Series([], dtype='object')
    serie_deceased_Daegu=pd.Series([], dtype='object')
    serie_confirmed_Incheon=pd.Series([], dtype='object')
    serie_released_Incheon=pd.Series([], dtype='object')
    serie_deceased_Incheon=pd.Series([], dtype='object')
    serie_confirmed_Gwangju=pd.Series([], dtype='object')
    serie_released_Gwangju=pd.Series([], dtype='object')
    serie_deceased_Gwangju=pd.Series([], dtype='object')
    serie_confirmed_Daejeon=pd.Series([], dtype='object')
    serie_released_Daejeon=pd.Series([], dtype='object')
    serie_deceased_Daejeon=pd.Series([], dtype='object')
    serie_confirmed_Ulsan=pd.Series([], dtype='object')
    serie_released_Ulsan=pd.Series([], dtype='object')
    serie_deceased_Ulsan=pd.Series([], dtype='object')
    serie_confirmed_Sejong=pd.Series([], dtype='object')
    serie_released_Sejong=pd.Series([], dtype='object')
    serie_deceased_Sejong=pd.Series([], dtype='object')
    serie_confirmed_Gyeonggi_do=pd.Series([], dtype='object')
    serie_released_Gyeonggi_do=pd.Series([], dtype='object')
    serie_deceased_Gyeonggi_do=pd.Series([], dtype='object')
    serie_confirmed_Gangwon_do=pd.Series([], dtype='object')
    serie_released_Gangwon_do=pd.Series([], dtype='object')
    serie_deceased_Gangwon_do=pd.Series([], dtype='object')
    serie_confirmed_Chungcheongbuk_do=pd.Series([], dtype='object')
    serie_released_Chungcheongbuk_do=pd.Series([], dtype='object')
    serie_deceased_Chungcheongbuk_do=pd.Series([], dtype='object')
    serie_confirmed_Chungcheongnam_do=pd.Series([], dtype='object')
    serie_released_Chungcheongnam_do=pd.Series([], dtype='object')
    serie_deceased_Chungcheongnam_do=pd.Series([], dtype='object')
    serie_confirmed_Jeollabuk_do=pd.Series([], dtype='object')
    serie_released_Jeollabuk_do=pd.Series([], dtype='object')
    serie_deceased_Jeollabuk_do=pd.Series([], dtype='object')
    serie_confirmed_Jeollanam_do=pd.Series([], dtype='object')
    serie_released_Jeollanam_do=pd.Series([], dtype='object')
    serie_deceased_Jeollanam_do=pd.Series([], dtype='object')
    serie_confirmed_Gyeongsangbuk_do=pd.Series([], dtype='object')
    serie_released_Gyeongsangbuk_do=pd.Series([], dtype='object')
    serie_deceased_Gyeongsangbuk_do=pd.Series([], dtype='object')
    serie_confirmed_Gyeongsangnam_do=pd.Series([], dtype='object')
    serie_released_Gyeongsangnam_do=pd.Series([], dtype='object')
    serie_deceased_Gyeongsangnam_do=pd.Series([], dtype='object')
    serie_confirmed_Jeju_do=pd.Series([], dtype='object')
    serie_released_Jeju_do=pd.Series([], dtype='object')
    serie_deceased_Jeju_do=pd.Series([], dtype='object')

    serie_test_time=pd.Series([], dtype='object')
    serie_negative_time=pd.Series([], dtype='object')
    serie_positive_time=pd.Series([], dtype='object')
    serie_confirmed_time=pd.Series([], dtype='object')
    serie_released_time=pd.Series([], dtype='object')
    serie_deceased_time=pd.Series([], dtype='object')

    serie_avg_temp=pd.Series([], dtype='object')
    serie_min_temp=pd.Series([], dtype='object')
    serie_max_temp=pd.Series([], dtype='object')
    serie_max_wind_speed=pd.Series([], dtype='object')
    serie_most_wind_direction=pd.Series([], dtype='object')
    serie_avg_relative_humidity=pd.Series([], dtype='object')

    medias_weather()

    for i in range(len(df_patInfo)):
        # Adicionei tabela region
        for j in range(len(df_region)):
            if df_patInfo["province"][i] == df_region["province"][j] and df_patInfo["city"][i] == df_region["city"][j]:
                serie_latitude[i]=df_region["latitude"][j]
                serie_longitude[i]=df_region["longitude"][j]
                serie_elementary[i]=df_region["elementary_school_count"][j]
                serie_kindergarten[i]=df_region["kindergarten_count"][j]
                serie_university[i]=df_region["university_count"][j]
                serie_academy[i]=df_region["academy_ratio"][j]
                serie_elderly[i]=df_region["elderly_population_ratio"][j]
                serie_elderly_alone[i]=df_region["elderly_alone_ratio"][j]
                serie_nursing[i]=df_region["nursing_home_count"][j]
        # Adicionei tabela SearchTrend
        for k in range(len(df_trend)):
            if df_patInfo["confirmed_date"][i] == df_trend["date"][k]:
                serie_cold[i]=df_trend["cold"][k]
                serie_flu[i]=df_trend["flu"][k]
                serie_pneumonia[i]=df_trend["pneumonia"][k]
                serie_coronavirus[i]=df_trend["coronavirus"][k]
        # Adicionei tabela TimeGender
        for l in range(0, len(df_tgender), 2):
            if df_patInfo["confirmed_date"][i] == df_tgender["date"][l]:

                if l > 0:
                    serie_deceased[i]=(df_tgender["deceased"][l] + df_tgender["deceased"][l + 1]) - (df_tgender["deceased"][l -1 ] + df_tgender["deceased"][l -2])
                    serie_confirmed[i]=(df_tgender["confirmed"][l] + df_tgender["confirmed"][l + 1]) - (df_tgender["confirmed"][l - 1] + df_tgender["confirmed"][l -2])
                    serie_deceased_male[i]=df_tgender["deceased"][l] - df_tgender["deceased"][l - 2]
                    serie_confirmed_male[i]=df_tgender["confirmed"][l] - df_tgender["confirmed"][l -2]
                    serie_deceased_female[i]=df_tgender["deceased"][l + 1] - df_tgender["deceased"][l - 1]
                    serie_confirmed_female[i]=df_tgender["confirmed"][l + 1] - df_tgender["confirmed"][l - 1]
                else:
                    serie_deceased[i]=df_tgender["deceased"][l] + df_tgender["deceased"][l + 1]
                    serie_confirmed[i]=df_tgender["confirmed"][l] + df_tgender["confirmed"][l + 1]
                    serie_deceased_male[i]=df_tgender["deceased"][l]
                    serie_confirmed_male[i]=df_tgender["confirmed"][l]
                    serie_deceased_female[i]=df_tgender["deceased"][l + 1]
                    serie_confirmed_female[i]=df_tgender["confirmed"][l + 1]

        # Adicionei tabela TimeAge
        for m in range(0, len(df_tage), 9):
            if df_patInfo["confirmed_date"][i] == df_tage["date"][m]:
                if m > 0:
                    serie_deceased_0[i]=df_tage["deceased"][m] - df_tage["deceased"][m -9]
                    serie_confirmed_0[i]=df_tage["confirmed"][m] - df_tage["confirmed"][m - 9]
                    serie_deceased_10[i]=df_tage["deceased"][m + 1] - df_tage["deceased"][m - 8]
                    serie_confirmed_10[i]=df_tage["confirmed"][m + 1] - df_tage["confirmed"][m -8]
                    serie_deceased_20[i]=df_tage["deceased"][m + 2] - df_tage["deceased"][m -7]
                    serie_confirmed_20[i]=df_tage["confirmed"][m + 2] - df_tage["confirmed"][m -7]
                    serie_deceased_30[i]=df_tage["deceased"][m + 3] - df_tage["deceased"][m -6]
                    serie_confirmed_30[i]=df_tage["confirmed"][m + 3] - df_tage["confirmed"][m -6]
                    serie_deceased_40[i]=df_tage["deceased"][m + 4] - df_tage["deceased"][m -5]
                    serie_confirmed_40[i]=df_tage["confirmed"][m + 4] - df_tage["confirmed"][m -5]
                    serie_deceased_50[i]=df_tage["deceased"][m + 5] - df_tage["deceased"][m -4]
                    serie_confirmed_50[i]=df_tage["confirmed"][m + 5] - df_tage["confirmed"][m -4]
                    serie_deceased_60[i]=df_tage["deceased"][m + 6] - df_tage["deceased"][m -3]
                    serie_confirmed_60[i]=df_tage["confirmed"][m + 6] - df_tage["confirmed"][m -3]
                    serie_deceased_70[i]=df_tage["deceased"][m + 7] - df_tage["deceased"][m -2]
                    serie_confirmed_70[i]=df_tage["confirmed"][m + 7] - df_tage["confirmed"][m -2]
                    serie_deceased_80[i]=df_tage["deceased"][m + 8] - df_tage["deceased"][m -1]
                    serie_confirmed_80[i]=df_tage["confirmed"][m + 8] - df_tage["confirmed"][m -1]
                else:
                    serie_deceased_0[i]=df_tage["deceased"][m]
                    serie_confirmed_0[i]=df_tage["confirmed"][m]
                    serie_deceased_10[i]=df_tage["deceased"][m + 1]
                    serie_confirmed_10[i]=df_tage["confirmed"][m + 1]
                    serie_deceased_20[i]=df_tage["deceased"][m + 2]
                    serie_confirmed_20[i]=df_tage["confirmed"][m + 2]
                    serie_deceased_30[i]=df_tage["deceased"][m + 3]
                    serie_confirmed_30[i]=df_tage["confirmed"][m + 3]
                    serie_deceased_40[i]=df_tage["deceased"][m + 4]
                    serie_confirmed_40[i]=df_tage["confirmed"][m + 4]
                    serie_deceased_50[i]=df_tage["deceased"][m + 5]
                    serie_confirmed_50[i]=df_tage["confirmed"][m + 5]
                    serie_deceased_60[i]=df_tage["deceased"][m + 6]
                    serie_confirmed_60[i]=df_tage["confirmed"][m + 6]
                    serie_deceased_70[i]=df_tage["deceased"][m + 7]
                    serie_confirmed_70[i]=df_tage["confirmed"][m + 7]
                    serie_deceased_80[i]=df_tage["deceased"][m + 8]
                    serie_confirmed_80[i]=df_tage["confirmed"][m + 8]
        # Adicionei tabela TimeProvince
        for n in range(0, len(df_tprovince), 17):
            if df_patInfo["confirmed_date"][i] == df_tprovince["date"][n]:
                serie_confirmed_Seoul[i]=df_tprovince["confirmed"][n]
                serie_released_Seoul[i]=df_tprovince["released"][n]
                serie_deceased_Seoul[i]=df_tprovince["deceased"][n]
                serie_confirmed_Busan[i]=df_tprovince["confirmed"][n + 1]
                serie_released_Busan[i]=df_tprovince["released"][n + 1]
                serie_deceased_Busan[i]=df_tprovince["deceased"][n + 1]
                serie_confirmed_Daegu[i]=df_tprovince["confirmed"][n + 2]
                serie_released_Daegu[i]=df_tprovince["released"][n + 2]
                serie_deceased_Daegu[i]=df_tprovince["deceased"][n + 2]
                serie_confirmed_Incheon[i]=df_tprovince["confirmed"][n + 3]
                serie_released_Incheon[i]=df_tprovince["released"][n + 3]
                serie_deceased_Incheon[i]=df_tprovince["deceased"][n + 3]
                serie_confirmed_Gwangju[i]=df_tprovince["confirmed"][n + 4]
                serie_released_Gwangju[i]=df_tprovince["released"][n + 4]
                serie_deceased_Gwangju[i]=df_tprovince["deceased"][n + 4]
                serie_confirmed_Daejeon[i]=df_tprovince["confirmed"][n + 5]
                serie_released_Daejeon[i]=df_tprovince["released"][n + 5]
                serie_deceased_Daejeon[i]=df_tprovince["deceased"][n + 5]
                serie_confirmed_Ulsan[i]=df_tprovince["confirmed"][n + 6]
                serie_released_Ulsan[i]=df_tprovince["released"][n + 6]
                serie_deceased_Ulsan[i]=df_tprovince["deceased"][n + 6]
                serie_confirmed_Sejong[i]=df_tprovince["confirmed"][n + 7]
                serie_released_Sejong[i]=df_tprovince["released"][n + 7]
                serie_deceased_Sejong[i]=df_tprovince["deceased"][n + 7]
                serie_confirmed_Gyeonggi_do[i]=df_tprovince["confirmed"][n + 8]
                serie_released_Gyeonggi_do[i]=df_tprovince["released"][n + 8]
                serie_deceased_Gyeonggi_do[i]=df_tprovince["deceased"][n + 8]
                serie_confirmed_Gangwon_do[i]=df_tprovince["confirmed"][n + 9]
                serie_released_Gangwon_do[i]=df_tprovince["released"][n + 9]
                serie_deceased_Gangwon_do[i]=df_tprovince["deceased"][n + 9]
                serie_confirmed_Chungcheongbuk_do[i]=df_tprovince["confirmed"][n + 10]
                serie_released_Chungcheongbuk_do[i]=df_tprovince["released"][n + 10]
                serie_deceased_Chungcheongbuk_do[i]=df_tprovince["deceased"][n + 10]
                serie_confirmed_Chungcheongnam_do[i]=df_tprovince["confirmed"][n + 11]
                serie_released_Chungcheongnam_do[i]=df_tprovince["released"][n + 11]
                serie_deceased_Chungcheongnam_do[i]=df_tprovince["deceased"][n + 11]
                serie_confirmed_Jeollabuk_do[i]=df_tprovince["confirmed"][n + 12]
                serie_released_Jeollabuk_do[i]=df_tprovince["released"][n + 12]
                serie_deceased_Jeollabuk_do[i]=df_tprovince["deceased"][n + 12]
                serie_confirmed_Jeollanam_do[i]=df_tprovince["confirmed"][n + 13]
                serie_released_Jeollanam_do[i]=df_tprovince["released"][n + 13]
                serie_deceased_Jeollanam_do[i]=df_tprovince["deceased"][n + 13]
                serie_confirmed_Gyeongsangbuk_do[i]=df_tprovince["confirmed"][n + 14]
                serie_released_Gyeongsangbuk_do[i]=df_tprovince["released"][n + 14]
                serie_deceased_Gyeongsangbuk_do[i]=df_tprovince["deceased"][n + 14]
                serie_confirmed_Gyeongsangnam_do[i]=df_tprovince["confirmed"][n + 15]
                serie_released_Gyeongsangnam_do[i]=df_tprovince["released"][n + 15]
                serie_deceased_Gyeongsangnam_do[i]=df_tprovince["deceased"][n + 15]
                serie_confirmed_Jeju_do[i]=df_tprovince["confirmed"][n + 16]
                serie_released_Jeju_do[i]=df_tprovince["released"][n + 16]
                serie_deceased_Jeju_do[i]=df_tprovince["deceased"][n + 16]

        # Adicionei tabela Time
        for o in range(len(df_time)):
            if df_patInfo["confirmed_date"][i] == df_time["date"][o]:
                if o > 0:
                    serie_test_time[i]=df_time["test"][o] - df_time["test"][o - 1]
                    serie_negative_time[i]=df_time["negative"][o] - df_time["negative"][o - 1]
                    serie_positive_time[i]=(df_time["test"][o] - df_time["negative"][o]) - (
                                df_time["test"][o - 1] - df_time["negative"][o - 1])
                    serie_confirmed_time[i]=df_time["confirmed"][o] - df_time["confirmed"][o - 1]
                    serie_released_time[i]=df_time["released"][o] - df_time["released"][o - 1]
                    serie_deceased_time[i]=df_time["deceased"][o] - df_time["deceased"][o - 1]
                elif o == 0:
                    serie_test_time[i]=df_time["test"][o]
                    serie_negative_time[i]=df_time["negative"][o]
                    serie_positive_time[i]=(df_time["test"][o] - df_time["negative"][o])
                    serie_confirmed_time[i]=df_time["confirmed"][o]
                    serie_released_time[i]=df_time["released"][o]
                    serie_deceased_time[i]=df_time["deceased"][o]

        for p in range(len(df_medias_weather)):
            if df_patInfo["province"][i] == df_medias_weather["province"][p] and \
                    df_patInfo["confirmed_date"][i] == df_medias_weather["date"][p]:
                serie_avg_temp[i]=df_medias_weather["avg_temp"][p]
                serie_min_temp[i]=df_medias_weather["min_temp"][p]
                serie_max_temp[i]=df_medias_weather["max_temp"][p]
                serie_max_wind_speed[i]=df_medias_weather["max_wind_speed"][p]
                serie_most_wind_direction[i]=df_medias_weather["most_wind_direction"][p]
                serie_avg_relative_humidity[i]=df_medias_weather["avg_relative_humidity"][p]

    df_patInfo.insert(len(df_patInfo.columns), "latitude", serie_latitude)
    df_patInfo.insert(len(df_patInfo.columns), "longitude", serie_longitude)
    df_patInfo.insert(len(df_patInfo.columns), "elementary_school_count", serie_elementary)
    df_patInfo.insert(len(df_patInfo.columns), "kindergarten_count", serie_kindergarten)
    df_patInfo.insert(len(df_patInfo.columns), "university_count", serie_university)
    df_patInfo.insert(len(df_patInfo.columns), "academy_ratio", serie_academy)
    df_patInfo.insert(len(df_patInfo.columns), "elderly_population_ratio", serie_elderly)
    df_patInfo.insert(len(df_patInfo.columns), "elderly_alone_ratio", serie_elderly_alone)
    df_patInfo.insert(len(df_patInfo.columns), "nursing_home_count", serie_nursing)

    df_patInfo.insert(len(df_patInfo.columns), "search_cold", serie_cold)
    df_patInfo.insert(len(df_patInfo.columns), "search_flu", serie_flu)
    df_patInfo.insert(len(df_patInfo.columns), "search_pneumonia", serie_pneumonia)
    df_patInfo.insert(len(df_patInfo.columns), "search_coronavirus", serie_coronavirus)

    df_patInfo.insert(len(df_patInfo.columns), "confirmed_cases", serie_confirmed)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_cases", serie_deceased)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_cases_male", serie_confirmed_male)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_cases_male", serie_deceased_male)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_cases_female", serie_confirmed_female)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_cases_female", serie_deceased_female)

    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_0s", serie_deceased_0)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_0s", serie_confirmed_0)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_10s", serie_deceased_10)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_10s", serie_confirmed_10)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_20s", serie_deceased_20)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_20s", serie_confirmed_20)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_30s", serie_deceased_30)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_30s", serie_confirmed_30)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_40s", serie_deceased_40)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_40s", serie_confirmed_40)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_50s", serie_deceased_50)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_50s", serie_confirmed_50)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_60s", serie_deceased_60)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_60s", serie_confirmed_60)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_70s", serie_deceased_70)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_70s", serie_confirmed_70)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_age_80s", serie_deceased_80)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_age_80s", serie_confirmed_80)

    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Seoul", serie_confirmed_Seoul)
    df_patInfo.insert(len(df_patInfo.columns), "released_Seoul", serie_released_Seoul)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Seoul", serie_deceased_Seoul)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Busan", serie_confirmed_Busan)
    df_patInfo.insert(len(df_patInfo.columns), "released_Busan", serie_released_Busan)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Busan", serie_deceased_Busan)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Daegu", serie_confirmed_Daegu)
    df_patInfo.insert(len(df_patInfo.columns), "released_Daegu", serie_released_Daegu)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Daegu", serie_deceased_Daegu)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Incheon", serie_confirmed_Incheon)
    df_patInfo.insert(len(df_patInfo.columns), "released_Incheon", serie_released_Incheon)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Incheon", serie_deceased_Incheon)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Gwangju", serie_confirmed_Gwangju)
    df_patInfo.insert(len(df_patInfo.columns), "released_Gwangju", serie_released_Gwangju)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Gwangju", serie_deceased_Gwangju)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Daejeon", serie_confirmed_Daejeon)
    df_patInfo.insert(len(df_patInfo.columns), "released_Daejeon", serie_released_Daejeon)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Daejeon", serie_deceased_Daejeon)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Ulsan", serie_confirmed_Ulsan)
    df_patInfo.insert(len(df_patInfo.columns), "released_Ulsan", serie_released_Ulsan)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Ulsan", serie_deceased_Ulsan)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Sejong", serie_confirmed_Sejong)
    df_patInfo.insert(len(df_patInfo.columns), "released_Sejong", serie_released_Sejong)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Sejong", serie_deceased_Sejong)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Gyeonggi_do", serie_confirmed_Gyeonggi_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Gyeonggi_do", serie_released_Gyeonggi_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Gyeonggi_do", serie_deceased_Gyeonggi_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Gangwon_do", serie_confirmed_Gangwon_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Gangwon_do", serie_released_Gangwon_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Gangwon_do", serie_deceased_Gangwon_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Chungcheongbuk_do", serie_confirmed_Chungcheongbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Chungcheongbuk_do", serie_released_Chungcheongbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Chungcheongbuk_do", serie_deceased_Chungcheongbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Chungcheongnam_do", serie_confirmed_Chungcheongnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Chungcheongnam_do", serie_released_Chungcheongnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Chungcheongnam_do", serie_deceased_Chungcheongnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Jeollabuk_do", serie_confirmed_Jeollabuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Jeollabuk_do", serie_released_Jeollabuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Jeollabuk_do", serie_deceased_Jeollabuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Jeollanam_do", serie_confirmed_Jeollanam_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Jeollanam_do", serie_released_Jeollanam_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Jeollanam_do", serie_deceased_Jeollanam_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Gyeongsangbuk_do", serie_confirmed_Gyeongsangbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Gyeongsangbuk_do", serie_released_Gyeongsangbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Gyeongsangbuk_do", serie_deceased_Gyeongsangbuk_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Gyeongsangnam_do", serie_confirmed_Gyeongsangnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Gyeongsangnam_do", serie_released_Gyeongsangnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Gyeongsangnam_do", serie_deceased_Gyeongsangnam_do)
    df_patInfo.insert(len(df_patInfo.columns), "confirmed_Jeju_do", serie_confirmed_Jeju_do)
    df_patInfo.insert(len(df_patInfo.columns), "released_Jeju_do", serie_released_Jeju_do)
    df_patInfo.insert(len(df_patInfo.columns), "deceased_Jeju_do", serie_deceased_Jeju_do)

    df_patInfo.insert(len(df_patInfo.columns), "tests_count", serie_test_time)
    df_patInfo.insert(len(df_patInfo.columns), "tests_negative", serie_negative_time)
    df_patInfo.insert(len(df_patInfo.columns), "tests_positive", serie_positive_time)
    df_patInfo.insert(len(df_patInfo.columns), "tests_confirmed", serie_confirmed_time)
    df_patInfo.insert(len(df_patInfo.columns), "tests_released", serie_released_time)
    df_patInfo.insert(len(df_patInfo.columns), "tests_deceased", serie_deceased_time)

    df_patInfo.insert(len(df_patInfo.columns), "avg_temp", serie_avg_temp)
    df_patInfo.insert(len(df_patInfo.columns), "min_temp", serie_min_temp)
    df_patInfo.insert(len(df_patInfo.columns), "max_temp", serie_max_temp)
    df_patInfo.insert(len(df_patInfo.columns), "max_wind_speed", serie_max_wind_speed)
    df_patInfo.insert(len(df_patInfo.columns), "most_wind_direction", serie_most_wind_direction)
    df_patInfo.insert(len(df_patInfo.columns), "avg_relative_humidity", serie_avg_relative_humidity)

    if not os.path.exists(os.getcwd() + "\\dataset_modificado\\"):
        os.mkdir(os.getcwd() + "\\dataset_modificado\\")

    df_patInfo=delete_column('infected_by', df_patInfo)
    df_patInfo=delete_column('contact_number', df_patInfo)
    df_patInfo=delete_column('deceased_date', df_patInfo)
    df_patInfo=delete_column('released_date', df_patInfo)
    df_patInfo=delete_column('symptom_onset_date', df_patInfo)
    df_patInfo=delete_column('patient_id', df_patInfo)
    df_patInfo=delete_rows(4730, df_patInfo)
    df_patInfo=delete_rows(4731, df_patInfo)
    df_patInfo=delete_rows(4732, df_patInfo)
    df_patInfo.reset_index(drop=True, inplace=True)

    df_patInfo = modificar_ficheiro(df_patInfo)

    df_patInfo.to_csv(os.getcwd() + "\\dataset_modificado\\PatientInfo_modificado.csv", index=False)


if __name__ == "__main__":
    data_selection()
    #df = pd.read_csv(os.getcwd() + "\\dataset_modificado\\PatientReg.csv")
    #print(df.isnull().sum())
    #print(df.count())