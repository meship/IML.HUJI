from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.array([0, 1, 2])
        self.mu_ = np.array([(1 / np.count_nonzero(y == i)) *
                             np.where(y.reshape((y.shape[0], 1)) == i, X, 0).sum(axis=0)
                             for i in self.classes_])
        cov_to_sum = [np.dot(np.reshape(X[i] - self.mu_[int(y[i])], (X[i].shape[0], 1)),
                             np.transpose(np.reshape(X[i] - self.mu_[int(y[i])], (X[i].shape[0], 1))))
                      for i in range(y.shape[0])]
        self.cov_ = (1 / (y.shape[0] - len(self.classes_)))* (np.array(cov_to_sum).sum(axis=0))
        self._cov_inv = inv(self.cov_)
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

        # result = np.zeros((len(self.classes_), X.shape[0]))
        def gaussian_calc(x, lda, fetchers, k):
            const = 1 / np.sqrt((2 * np.pi) ** fetchers * np.linalg.det(self.cov_))
            return const * np.e ** (-0.5 * ((x - lda.mu_[k]).T).dot(lda._cov_inv).dot((x - lda.mu_[k])))

        func = lambda t, i: self.pi_[i] * gaussian_calc(t, self, X.shape[1], i)
        result = np.array([np.apply_along_axis(func, 1, X, i) for i in range(len(self.classes_))])
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
