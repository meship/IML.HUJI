from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.array([0, 1, 2])
        self.mu_ = np.array([(1 / (np.count_nonzero(y == i))) *
                             np.where(y.reshape((y.shape[0], 1)) == i, X, 0).sum(axis=0)
                             for i in self.classes_])
        self.vars_ = np.array([((X[y == self.classes_[i]] - self.mu_[i]) ** 2).sum(axis=0) /
                               (X[y == self.classes_[i]].shape[0] - 1) for i in range(len(self.classes_))])
        self.pi_ = np.array([np.count_nonzero(y == i) / y.shape[0] for i in self.classes_])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        like = self.likelihood(X)
        return np.array([self.classes_[num] for num in np.argmax(like, axis=1)])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        def gaussian_calc(x, gnb, fetchers, k, cov):
            const = 1 / np.sqrt((2 * np.pi) ** fetchers * np.linalg.det(cov))
            return const * np.e ** (-0.5 * (x - gnb.mu_[k]).T.dot(inv(cov)).dot((x - gnb.mu_[k])))

        base_func = lambda t, i: self.pi_[i] * gaussian_calc(t, self, X.shape[1], i, np.diag(self.vars_[i]))
        result = np.array([np.apply_along_axis(base_func, 1, X, i) for i in range(len(self.classes_))])
        return result.T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))

