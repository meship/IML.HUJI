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
    Data_split = np.split(X, cv)
    label_split = np.split(y, cv)
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



