import pandas as pd
import numpy as np
import os
from arff2csv import arff2csv


def fix_files():
    for gadget in ["phone", "watch"]:
        os.chdir(gadget)
        for instrument in ["accel", "gyro"]:
            os.chdir(instrument)
            for subject in range(1600, 1651):
                filename = "data_%d_%s_%s" % (subject, instrument, gadget)
                try:
                    arff2csv(filename + ".arff", filename + ".csv")
                except:
                    pass
            os.chdir("RP")
        os.chdir("RP")


def load_data(gadget, instrument):
    os.chdir("wisdm-dataset/arff_files")
    if not os.path.exists("%s/%s/data_1610_%s_%s.csv" % (gadget, instrument, instrument, gadget)):
        fix_files()
    try:
        df = pd.read_csv("merged_dataset_%s_%s.csv" % (instrument, gadget), index_col=0)
    except:
        df = pd.DataFrame()
        os.chdir(gadget)
        os.chdir(instrument)
        for subject in range(1600, 1651):
            filename = "data_%d_%s_%s.csv" % (subject, instrument, gadget)
            try:
                df = df.append(pd.read_csv(filename))
            except:
                pass
        os.chdir("RP")
        os.chdir("RP")
        df = normalize(df)
        df.to_csv("merged_dataset_%s_%s.csv" % (instrument, gadget))
    os.chdir("")
    return df


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != "ACTIVITY" and feature_name != "class":
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def get_activities_key():
    with open("wisdm-dataset/activity_key.txt", 'r') as f:
        lines = f.readlines()
        lines = [line.split("=") for line in lines]
        return {line[1].strip(): line[0].strip() for line in lines}


def load_scenario(scenario, gadget, instrument):
    if scenario.upper() == "A":
        try:
            df = pd.read_csv("wisdm-dataset/arff_files/dataset_scenarioA_%s_%s.csv" % (instrument, gadget), index_col=0)
        except:
            df = load_data(gadget, instrument)
            df = df.replace(["A", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "O", "P", "Q", "R", "S"], 0)
            df = df.replace("B", 1)
            df.to_csv("wisdm-dataset/arff_files/dataset_scenarioA_%s_%s.csv" % (instrument, gadget))
    elif scenario.upper() == "B":
        try:
            df = pd.read_csv("wisdm-dataset/arff_files/dataset_scenarioB_%s_%s.csv" % (instrument, gadget), index_col=0)
        except:
            df = load_data(gadget, instrument)
            df = df.replace(["A", "B", "C", "D", "E", "M"], 0)
            df = df.replace(["F", "G", "O", "P", "Q", "R", "S"], 1)
            df = df.replace(["H", "I", "J", "K", "L"], 2)
            df.to_csv("wisdm-dataset/arff_files/dataset_scenarioB_%s_%s.csv" % (instrument, gadget))
    else:
        try:
            df = pd.read_csv("wisdm-dataset/arff_files/dataset_scenarioC_%s_%s.csv" % (instrument, gadget), index_col=0)
        except:
            df = load_data(gadget, instrument)
            letters = [chr(letter) for letter in range(ord('A'), ord('S') + 1)]
            letters.remove('N')
            values = [i for i in range(len(letters))]
            for letter, value in zip(letters, values):
                df = df.replace(letter, value)
            df.to_csv("wisdm-dataset/arff_files/dataset_scenarioC_%s_%s.csv" % (instrument, gadget))
    return df


def get_data(scenario, gadget, instrument):
    df = load_scenario(scenario, gadget, instrument)
    df = df.dropna(axis=1)
    X = df.drop(["ACTIVITY", "class"], axis=1)
    y = np.array(df["ACTIVITY"])
    return X, y
