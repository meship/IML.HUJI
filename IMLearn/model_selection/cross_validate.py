from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    Data_split = np.array_split(X, cv)
    label_split = np.array_split(y, cv)
    train_error = 0
    validation_error = 0
    for i in range(cv):
        train_X = np.concatenate(np.delete(Data_split,i,axis=0))
        train_y = np.concatenate(np.delete(label_split,i,axis=0))
        estimator.fit(train_X, train_y)
        pred_y_validation = estimator.predict(Data_split[i])
        pred_y_train = estimator.predict(train_X)
        train_error += scoring(train_y, pred_y_train)
        validation_error += scoring(label_split[i], pred_y_validation)
    return train_error / cv, validation_error / cv

    # validation_error = []
    # train_error = []
    # check_X = np.array_split(X, cv)
    # check_y = np.array_split(y, cv)
    # for fold_index in range(cv):
    #     # ***************************
    #     # a = X[0:fold_index * chunk_size]
    #     # b = X[(fold_index + 1) * chunk_size:]
    #     # k_train_x = np.concatenate((a, b), axis=0)
    #     # k_test_x = X[fold_index * chunk_size: (fold_index + 1) * chunk_size]
    #     #
    #     # c = y[0:fold_index * chunk_size]
    #     # d = y[(fold_index + 1) * chunk_size:]
    #     # k_train_y = np.concatenate((c, d), axis=0)
    #     # k_test_y = y[fold_index * chunk_size: (fold_index + 1) * chunk_size]
    #     # print(f"TrainX is: {k_test_x}\nTrainY is: {k_test_y}")
    #     # print(f"TrainX is: {k_train_x}\nTrainY is: {k_train_y}")
    #     # print(f"train size is:{k_train_y.shape[0]} and test size is: {k_test_y.shape[0]}")
    #
    #     # *************************
    #     k_train_x = None
    #     for index in range(cv):
    #         if index != fold_index:
    #             if k_train_x is None:
    #                 k_train_x = check_X[index]
    #             else:
    #                 k_train_x = np.concatenate((k_train_x, check_X[index]), axis=0)
    #         else:
    #             k_test_x = check_X[index]
    #
    #     k_train_y = np.array([])
    #     for index in range(cv):
    #         if index != fold_index:
    #             k_train_y = np.concatenate((k_train_y, check_y[index]), axis=0)
    #         else:
    #             k_test_y = check_y[index]
    #
    #     # print(f"TrainX is: {k_test_x}\nTrainY is: {k_test_y}")
    #     # print(f"TrainX is: {k_train_x}\nTrainY is: {k_train_y}")
    #     estimator.fit(k_train_x, k_train_y)
    #     validation_error.append(scoring(estimator.predict(k_test_x), k_test_y))
    #     train_error.append(scoring(estimator.predict(k_train_x), k_train_y))
    #
    # return np.mean(train_error), np.mean(validation_error)





