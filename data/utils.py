"""
Data Set Utils

**In progress - Incomplete **

This will be a collection of static methods to help manage a dataset

"""

__author__ = "Jon Wiggins"

import math
import numpy as np
import pandas as pd
import itertools


def jackknife(data, fold_count, unify_train, save, save_dir):
    """
    Jackknifes the given dataframe into training and testing sets
    Can be saved to a directory and the training sets can be unified to seperate

    :param data: a dataframe source
    :param fold_count: the number of folds to create
    :param unify_train: if true, returns 1 training dataframe holding all folds, else returns a list of dataframes
    :param save: if true, saves the folds to the given directory
    :param save_dir: a directory as a string on where to save the folds as a csv

    :return: training folds, one dataframe if unified, a list of dataframes if not
    :return: a testing fold as a dataframe
    """
    folds = np.array_split(data, fold_count)
    np.random.shuffle(folds)
    test_fold = folds[0]

    if not unify_train:
        if save:
            for count, fold in enumerate(folds[1:]):
                fold.to_csv(save_dir + "train" + str(count) + ".csv")
            test_fold.to_csv(save_dir + "test.csv")
        return folds[1:], test_fold

    train_fold = folds[1]
    for fold in folds[2:]:
        train_fold = train_fold.append(fold)

    if save:
        train_fold.to_csv(save_dir + "train.csv")
        test_fold.to_csv(save_dir + "test.csv")
    return train_fold, test_fold


def enumerate_hyperparameter_combinations(parameter_to_options):
    """
    Returns a list of dictionaries of all hyperparameter options

    :param parameter_to_options: a dictionary that maps parameter name to a list of possible values

    :return: a list of dictionaries that map parameter names to set values
    """
    keys, values = zip(*parameter_to_options.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def test_model_accuracy(model, probe_method, test_set, test_answer):
    """
    Tests the model on the given set and returns the accuracy

    :param model: a trained model
    :param probe_method: the models query method, takes an instance as a parameter and returns the prediction
    :param test_set: an iterable object to probe with
    :param test_answer: an iterable object to correlate with the models predictions

    :return: the number of correct predicitons / total number of probes
    """
    correct_count = 0
    for index, element in enumerate(test_set):
        if model.probe_method(element) == test_answer[index]:
            correct_count += 1

    return correct_count / len(test_set)


def test_model_error(model, probe_method, test_set, test_answer):
    """
    Tests the model on the given set and returns the error

    :param model: a trained model
    :param probe_method: the models query method, takes an instance as a parameter and returns the prediction
    :param test_set: an iterable object to probe with
    :param test_answer: an iterable object to correlate with the models predictions

    :return: 1- the number of correct predicitons / total number of probes
    """
    return 1 - test_model_accuracy(model, probe_method, test_set, test_answer)


def get_average_accuracy_and_sd(model, probe_method, folds):
    # TODO generalize this for all different models
    accs = []
    for test_fold in folds:
        if test_fold == folds[0]:
            train_set = folds[1]
        else:
            train_set = folds[0]
        for element in folds:
            if element == test_fold:
                continue
            train_set = train_set.append(element)
        # TODO generalize
        accs.append(
            test_model_accuracy(
                model, probe_method, test_set=test_fold, test_answer=test_fold
            )
        )
    return np.mean(accs, axis=0), np.std(accs, axis=0)


def grid_search(
    model_type,
    probe_method,
    parameter_to_options,
    train_set,
    train_answers,
    test_set,
    test_answers,
    fold_count,
    print_results,
):
    # TODO generalize this for all different models
    best_params = None
    best_model = None
    best_acc = None
    for parameter_set in enumerate_hyperparameter_combinations(parameter_to_options):
        current_model = model_type(parameter_set)
        current_model.train(train_set, train_answers)
        acc, sd = get_average_accuracy_and_sd(current_model, probe_method, fold_count)
        if print_results:
            print(type(model_type))
            print("Parameters:", parameter_set)
            print("Average Accuracy:", acc)
        if not best_acc or best_acc < acc:
            best_model = current_model
            best_params = parameter_set
            best_acc = acc

    return best_model, best_params, best_acc

