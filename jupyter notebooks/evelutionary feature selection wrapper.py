import os
from copy import deepcopy
import random
import numpy as np
from data_initialization import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit


def initialize_population(pop_size, n_feat=91):
    return [[[True if v == i % n_feat else False for v in range(n_feat)], 0] for i in range(pop_size)]  # each individual has only 1 feature
    # return [[[bool(v) for v in np.random.randint(2, size=n_feat)], 0] for _ in range(pop_size)] # random initialization


def evaluate_fitness(ind, X, y):
    if ind[0].count(True) == 0:
        ind[1] = -1
    else:
        X = X.iloc[:, ind[0]]
        ind[1] = np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=StratifiedShuffleSplit(n_splits=3, test_size=0.7))) - 0.1 / 91 * ind[0].count(True)


# def crossover(a, b, n_feat=91):
#     ix = [random.randint(0, n_feat) for _ in range(2)]
#     ix.sort()
#     a, b = deepcopy(a), deepcopy(b)
#     a[0][ix[0]:ix[1]], b[0][ix[0]:ix[1]] = b[0][ix[0]:ix[1]], a[0][ix[0]:ix[1]]
#     return [a, b]

def crossover(a, b, n_feat=91, prob=0.1):
    ix = [i for i in range(91) if a[0][i] or b[0][i]]
    a, b = deepcopy(a), deepcopy(b)
    for i in ix:
        if random.random() < prob:
            a[0][i], b[0][i] = b[0][i], a[0][i]
    return [a, b]


def mutate(ind, prob=0.02):
    gene = [v if random.random() > prob else not v for v in ind[0]]
    return [gene, 0]


def run(pop_size, n_generations, X, y, elitism=0.02, tournament_size=0.1, diversity_threshold=0.001, verbose=False):
    # initialize
    elitism = round(elitism * pop_size)
    tournament_size = round(tournament_size * pop_size)
    pop = initialize_population(pop_size)
    best = pop[0]
    best_gen = 0
    if verbose:
        print("Initializing genetic algorithm.\nPopulation: %d\nNumber of generations: %d\nTournament size: %d\nElitism size: %d\n" % (pop_size, n_generations, tournament_size, elitism))
    # evolve population
    for gen in range(n_generations):
        if verbose:
            print("Evaluating generation %d\n" % gen)

        # rank population
        for ind in pop:
            evaluate_fitness(ind, X, y)
        pop.sort(reverse=True, key=lambda x: x[1])
        if verbose:
            print("Best generational individual fitness: %f" % pop[0][1])
            print("Number of features: %d" % (pop[0][0].count(True)))
            print(pop[0][0], end="\n\n")
            print("Worst generational individual fitness: %f" % pop[-1][1])
            print("Number of features: %d" % (pop[-1][0].count(True)))
            print(pop[-1][0], end="\n\n")

        # update best solution
        if best[1] < pop[0][1]:
            best = pop[0]
            best_gen = gen

        # restart if lost diversity
        std = np.std([i[1] for i in pop])
        if std < diversity_threshold:
            pop = initialize_population(pop_size)
            if verbose:
                print("Fitness standard deviation: %f. Lack of diversity, restarting." % std)
            continue
        print("\n")

        # apply elitism
        new_pop = [pop[i] for i in range(elitism)]

        # do tournament selection with 2 winners, crossover to generate 2 children, mutate parents
        while len(new_pop) < pop_size:
            tournament = random.sample(pop, tournament_size)
            tournament.sort(reverse=True, key=lambda x: x[1])
            new_pop.extend(crossover(tournament[0], tournament[1]))
            new_pop.extend([mutate(tournament[0]), mutate(tournament[1])])

        # if population > pop_size cutoff extras
        pop = new_pop[:pop_size]

    # check last generation and return best overall
    for ind in pop:
        evaluate_fitness(ind, X, y)
    pop.sort(reverse=True, key=lambda x: x[1])
    # update best solution
    if best[1] < pop[0][1]:
        best = pop[0]
        best_gen = n_generations
        if verbose:
            print("Found new best overall individual on this generation.")
    return best, best_gen


if __name__ == "__main__":
    os.chdir("..")
    X, y = get_data()
    best, gen = run(91, 50, X, y, verbose=True)
    print("Found best individual at generation %d with fitness %f" % (gen, best[1]))
    print(best)
    X = X.iloc[:, best[0]]
    print(np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.5))))
