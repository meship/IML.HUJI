from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils import *
pio.templates.default = "simple_white"


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
    return data_array[:,0:2], np.reshape(data_array[:,2:], (data_array[:,2:].shape[0]))

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perc = Perceptron().fit(X, y)
        losses = perc.loss_array

        # Plot figure
        go.Figure([go.Scatter(x=list(range(len(losses)+1)), y=losses, mode='lines', name=r'$\widehat\sigma^2$'),],
                  # layout=go.Layout(title=r"$\text{(6) Estimation of Variance As Function Of Number Of Samples}$",
                  #                  xaxis_title="$m\\text{ - number of samples}$",
                  #                  yaxis_title="r$\hat\sigma^2$")
                        ).show()



def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        from IMLearn.metrics import accuracy
        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        lda_pred = lda.predict(X)
        lda_acc = accuracy(y, lda_pred)
        gnb = GaussianNaiveBayes().fit(X,y)
        gnb_pred = gnb.predict(X)
        gbn_acc = accuracy(y, lda_pred)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        model_names = ["", ""]
        title = ""
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)

        models = [lda_pred, gnb_pred]
        symbols = class_symbols
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y, symbol=symbols[y], colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))], rows=1, cols=i + 1)

        for i in range(len(lda.classes_)):
            fig.add_traces([])




if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
