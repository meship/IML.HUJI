import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(lambda: DecisionStump(), n_learners).fit(train_X, train_y)
    test_error = [adaBoost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
    train_errors = [adaBoost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    fig = go.Figure(
        layout=go.Layout(title=
                         rf"$\textbf{{training- and test errors as a function of the number of fitted learners}}$",
                         margin=dict(t=100)))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_error, mode='lines', name="test_error"))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, mode='lines', name="train_error"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # predictions = [adaBoost.partial_predict(test_X, t) for t in T]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        predict = lambda X: adaBoost.partial_predict(X, t)
        fig.add_traces([decision_surface(predict, lims[0], lims[1]),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Adaboost prediction over 5, 50 100 and 250 Iterations with {noise} noise}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    best_fit = np.argmin(np.array(test_error))
    predict = lambda X: adaBoost.partial_predict(X, best_fit)
    acc = accuracy(test_y, adaBoost.partial_predict(test_X, best_fit))
    fig = go.Figure([decision_surface(predict, lims[0], lims[1]),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])

    fig.update_layout(title=
    rf"$\textbf{{Adaboost prediction over {best_fit} iterations and {acc} accuracy train error with {noise} noise}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    new_D = 5 * adaBoost.D_ / adaBoost.D_[np.argmax(adaBoost.D_)]
    fig = go.Figure([decision_surface(adaBoost.predict, lims[0], lims[1]),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                            size=new_D,
                                            line=dict(color="black", width=1)))])
    fig.update_layout(title=rf"$\textbf{{Decision surface with weighted samples with {noise} noise}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
