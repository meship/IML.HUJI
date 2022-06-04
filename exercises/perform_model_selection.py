from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    set_for_x = np.linspace(-1.2,2,num=n_samples)
    f = np.vectorize(lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2))
    noise_less_data = f(set_for_x)
    noises = np.random.normal(0, noise, n_samples)
    data_with_noise = noise_less_data + noises
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(set_for_x), pd.Series(data_with_noise), 0.66)
    X_train = X_train.to_numpy().reshape(X_train.shape[0])
    y_train = np.array(y_train)
    X_test = X_test.to_numpy().reshape(X_test.shape[0])
    y_test = np.array(y_test)

    fig = go.Figure([go.Scatter(x=set_for_x, y=noise_less_data, name="Real Model",  mode='markers+lines',   marker=dict(
            color='LightSkyBlue'),  showlegend=True)])
    fig.add_traces([go.Scatter(x=X_test, y=y_test, name="test Model",  mode='markers', marker=dict(
            color='orchid'), showlegend=True)])
    fig.add_traces([go.Scatter(x=X_train, y=y_train, name="train Model", mode='markers', marker=dict(
            color='teal'), showlegend=True)])
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = []
    val_errors = []
    for k in range(1,11):
        polyFit = PolynomialFitting(k)
        train_error, val_error = cross_validate(polyFit, X_train, y_train, mean_square_error)
        train_errors.append(train_error)
        val_errors.append(val_error)
    train_errors = np.array(train_errors)
    val_errors = np.array(val_errors)
    fig1 = go.Figure([go.Scatter(x=list(range(1,11)), y=train_errors, name="Real Model", mode='lines', line=dict(
        color='limegreen'), showlegend=True)])
    fig1.add_traces([go.Scatter(x=list(range(1,11)), y=val_errors, name="test Model", mode='lines', line=dict(
        color='mediumaquamarine'), showlegend=True)])
    fig1.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(val_errors)
    polyFit = PolynomialFitting(k).fit(X_train, y_train)
    loss = polyFit.loss(X_test, y_test)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
