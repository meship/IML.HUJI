from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()
    features = full_data[[
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
        "sqft_above",
        "sqft_basement",]]
    features["age"] = full_data["yr_built"].apply(lambda t: 2022-t)
    features["renovated"] = np.where(full_data["yr_renovated"] == 0,0,1)
    features["grade"] = np.where(np.isnan(full_data["grade"]),0,full_data["grade"])

    prices = np.where(np.isnan(full_data["price"]),0,full_data["price"])
    return features, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    function = lambda t: np.cov(t,y)[0][1] / np.sqrt(np.var(t)*np.var(y))
    corr_array = X.apply(function, axis=0)
    corr_data = {'features': [x for x in corr_array._info_axis], 'corr': [x for x in corr_array]}
    corr_data_frame = pd.DataFrame(data=corr_data)
    fig = px.bar(corr_data_frame, x='features', y='corr',range_y=[-1,1])
    fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, labeles = load_data("../datasets/house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, labeles)

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
