from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px
from IMLearn.metrics import accuracy
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """

    data_array = np.load(filename)
    return data_array[:, 0:2], np.reshape(data_array[:, 2:], (data_array[:, 2:].shape[0]))


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perc = Perceptron().fit(X, y)
        losses = perc.loss_array


        # Plot figure
        go.Figure([go.Scatter(x=list(range(len(losses) + 1)), y=losses, mode='lines', name=r'$\widehat\sigma^2$'), ],
                  layout=go.Layout(title=r"$\text{Perceptron loss over the data as a function of iterations}$",
                                   xaxis_title="$\\text{number of iterations}$",
                                   yaxis_title="$\\text{loss}$")
                  ).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        gnb_pred, lda_pred = gnb.predict(X), lda.predict(X)
        gbn_acc, lda_acc = accuracy(y, gnb_pred), accuracy(y, lda_pred)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        model_names = ["Gaussian Naive Bayes classifier with accuracy %.3f" % gbn_acc,
                       "LDA classifierwith accuracy %.3f" % lda_acc]
        models = [gnb_pred, lda_pred]
        color = np.array(px.colors.qualitative.Safe)
        symbols = class_symbols
        preds = [gnb_pred.astype(int), lda_pred.astype(int)]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        fig.update_layout(title=rf"$\textbf{{Predicted Classifiers Over {f} Dataset}}$",
                          margin=dict(t=100), showlegend=False)

        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=color[preds[i]], symbol=symbols[y.astype(int)],
                                                   line=dict(color="black", width=1),
                                                   colorscale=[custom[0], custom[-1]]
                                                   ))], rows=1, cols=i + 1)

        for i in range(len(lda.classes_)):
            fig.add_traces([get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))], rows=1, cols=1)
            fig.add_traces([go.Scatter(x=[gnb.mu_[i][0]], y=[gnb.mu_[i][1]],
                                       marker=dict(symbol=symbols[1], color="black", size=20))], rows=1, cols=1)
            fig.add_traces([get_ellipse(lda.mu_[i], lda.cov_)], rows=1, cols=2)
            fig.add_traces([go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]],
                                       marker=dict(symbol=symbols[1], color="black", size=20))], rows=1, cols=2)
        fig.show()

        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
