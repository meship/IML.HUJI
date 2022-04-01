import math

from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
HEAT_MAP_TITLE = "the Log_Likelihood of MV Gaussian With Different Mean and Same Cov Regarding Given Samples"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10,1,1000)
    unigaussian = UnivariateGaussian(False)
    unigaussian.fit(samples)
    print(unigaussian.mu_,unigaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    mu, sigma = 10, 1
    estimated_mean = []
    for m in ms:
        X = samples[:m]
        ug = UnivariateGaussian(False)
        ug.fit(X)
        estimated_mean.append(np.abs(ug.mu_-mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$'),],
              layout=go.Layout(title=r"$\text{Absolute Distance Between the Estimated- and True Value "
                                     r"of the Expectation,as a Function of the Sample Size "
                                     r"of 1000 iid∼ N (10,1) Samples}$",
                               xaxis_title="number of samples",
                               yaxis_title="absolute distance",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    Y = unigaussian.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=Y, mode='markers', name=r'$\widehat\mu$'), ],
              layout=go.Layout(title=r"$\text{Estimation of Pdf Function on the Given Samples From the Estimate Mean "
                                     r"and Variance of 1000 iid∼ N (10,1) Samples}$",
                               xaxis_title="samples value",
                               yaxis_title="estimated pdf returned value",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0],[0, 0, 1, 0],[0.5, 0, 0, 1]])
    num_of_sump = 1000
    samples = np.random.multivariate_normal(mu, cov, num_of_sump)
    mul_gaussian = MultivariateGaussian()
    mul_gaussian.fit(samples)
    print(mul_gaussian.mu_)
    print(mul_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10,10,200)
    f3 = np.linspace(-10,10,200)
    log_like_mat = np.empty([200,200])
    max = float('-inf')
    for index_i , row_val in enumerate(f1):
        for index_j, col_val in enumerate(f3):
            new_mu = np.array([row_val, 0, col_val, 0])
            temp = MultivariateGaussian.log_likelihood(new_mu, cov, samples)
            log_like_mat[index_i][index_j] = temp
            if max<temp:
                max = temp
                max_i, max_j = index_i, index_j

    fig = go.Figure(data=go.Heatmap(x=f1, y=f3,z=log_like_mat, reversescale=True, xaxis='x', yaxis='y'))
    fig.update_layout(xaxis_title="${f1}$",
                      yaxis_title="${f3}$",title=HEAT_MAP_TITLE)

    fig.show()


    # Question 6 - Maximum likelihood
    print("the model with the max likelihood is [%.3f, 0, %.3f, 0]" % (f1[max_i], f3[max_j]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
