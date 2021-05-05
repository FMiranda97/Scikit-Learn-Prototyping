import os
from copy import deepcopy
import random
import numpy as np
import sklearn.linear_model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data_initialization import get_data_from_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

__author__ = 'Ernesto Costa'
__date__ = 'April 2021'

import math
import operator
import random
import matplotlib.pyplot as plt


def s_pbil(pop_size, cromo_size, sample_size, l_rate, fitness, numb_gener):
    # for statistics
    best_list = []
    average_list = []
    best_pop = None
    # Initial probability model
    prob_vector = [0.5 for _ in range(cromo_size)]
    for gen in range(numb_gener):
        print("Generation %d / %d" % (gen, numb_gener))
        # Generate population using the probability dristibution
        sol_vectors = [[gener_vector(prob_vector), 0] for _ in range(pop_size)]
        eval_vectors = [[sol_vectors[count][0], fitness(sol_vectors[count][0])] for count in range(pop_size)]
        # Reduce to the sample size
        eval_vectors.sort(key=operator.itemgetter(1), reverse=True)
        sample_vectors = eval_vectors[:sample_size]
        if best_pop is None or best_pop[1] < sample_vectors[0][1]:
            best_pop = sample_vectors[0]
        best_list.append(best_pop[1])
        average_list.append(sum([indiv[1] for indiv in sample_vectors]) / len(sample_vectors))
        sample_chromo = [indiv[0] for indiv in sample_vectors]
        # compute the frequency of the alleles
        new_prob_vector = [sum(allele) / len(allele) for allele in list(zip(*sample_chromo))]
        # print(new_prob_vector)
        for i in range(cromo_size):
            prob_vector[i] = prob_vector[i] * (1.0 - l_rate) + new_prob_vector[i] * l_rate
        print(best_pop[1], best_pop[0])
    return best_pop, best_list, average_list


def gener_vector(prob_vector):
    """ from the probability vector to a binary string."""
    return [1 if random.random() < x else 0 for x in prob_vector]





def display(data_best, data_average, function_name):
    plt.title('Estimation Distribution Algorithm: PBIL')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    x = range(1, len(data_best) + 1)
    plt.plot(x, data_best, label='Best')
    plt.plot(x, data_average, label='Average')
    plt.legend(loc='best')
    plt.show()


def run(numb_runs, pop_size, cromo_size, sample_size, l_rate, fitness, numb_gener, title='DT'):
    all_best = []
    for count in range(numb_runs):
        print("Run %d / %d" % (count, numb_runs))
        best_pop, best_list, average_list = s_pbil(pop_size, cromo_size, sample_size, l_rate, fitness, numb_gener)
        all_best.append(best_list)
    all_best_gen = zip(*all_best)
    average_all_best_gen = [sum(values_gen) / numb_runs for values_gen in all_best_gen]
    plt.title('PBIL for feature selection using %s' % title)
    plt.xlabel('Generation')
    plt.ylabel('Average Best Fitness')
    x = range(1, numb_gener + 1)
    plt.plot(x, average_all_best_gen, 'r-o')
    plt.show()


def evaluate_fitness(X, y, model):
    def evaluate(vector):
        vector = [True if x == 1 else False for x in vector]
        if vector.count(True) == 0:
            return -1
        else:
            X_selected = X.iloc[:, vector]
            return np.min(cross_val_score(model, X_selected, y, cv=StratifiedShuffleSplit(n_splits=3, test_size=0.3), n_jobs=-1))

    return evaluate


if __name__ == '__main__':
    X, y = get_data_from_csv('../dataset/PatientInfo_Final.csv')
    pop_size = 100
    cromo_size = X.shape[1]
    percent = 0.02
    sample_size = int(percent * pop_size)
    l_rate = 0.05
    numb_gener = 200
    numb_runs = 1

    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        LinearDiscriminantAnalysis()
    ]
    for classifier in classifiers:

        fitness = evaluate_fitness(X, y, classifier)
        print(str(classifier))
        run(numb_runs, pop_size, cromo_size, sample_size, l_rate, fitness, numb_gener, title=str(classifier))
