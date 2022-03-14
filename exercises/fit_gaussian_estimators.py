import math

from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import norm
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    s = np.random.normal(10,1,1000)
    unigaussian = UnivariateGaussian(False)
    unigaussian.fit(s)
    print(unigaussian.mu_,unigaussian.var_)


    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    mu, sigma = 10, 1
    estimated_mean = []
    for m in ms:
        X = s[:m]
        ug = UnivariateGaussian(False)
        ug.fit(X)
        estimated_mean.append(np.abs(ug.mu_-mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$'),],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    Y = unigaussian.pdf(s)
    go.Figure([go.Scatter(x=s, y=Y, mode='markers', name=r'$\widehat\mu$'), ],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()


def test_multivariate_gaussian():
    pass
    # Question 4 - Draw samples and print fitted model
    #raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    #raise NotImplementedError()

    # Question 6 - Maximum likelihood
    #raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
