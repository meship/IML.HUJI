from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error

import sys

sys.path.append("../")
from utils import *

np.random.seed(17)
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
    full_data = full_data.drop(full_data[full_data['price'] <= 0].index)
    features = full_data[[
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
        "sqft_above",
        "sqft_basement",
        "view",
        "waterfront",
        "grade"
    ]]
    features["age"] = full_data["yr_built"].apply(lambda t: 2022 - t)
    features["renovated"] = np.where(full_data["yr_renovated"] == 0, 0, 1)
    return features, full_data['price']


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
    function = lambda t: np.cov(t, y)[0][1] / np.sqrt(np.var(t) * np.var(y))
    corr_array = X.apply(function, axis=0)
    price_by_sqft_living = {'sqft_living': X['sqft_living'], 'Price': y}
    price_by_sqft_living_df = pd.DataFrame(data=price_by_sqft_living)
    fig = px.scatter(price_by_sqft_living_df, x="sqft_living", y="Price",
                     title='Price as a Function of sqft_living With '
                           'Pearson Corr of: %0.4f' % (corr_array['sqft_living']))
    pio.write_html(fig, output_path + 'corr_living.html')

    price_by_age = {'House_age': X['age'], 'Price': y}
    price_by_age_df = pd.DataFrame(data=price_by_age)
    fig = px.scatter(price_by_age_df, x="House_age", y="Price", range_x=[0, 150],
                     title='Price as a Function of the \'House Age\' With '
                           'Pearson Corr of: %0.4f' % (corr_array['age']))
    pio.write_html(fig, output_path + 'corr_age.html')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, labels)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, labels)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    l_r = LinearRegression()
    average_loss = np.empty([91])
    y_upper = []
    y_lower = []
    var = np.empty([91])
    for i in range(10, 101):
        loss_10_semp = np.empty([10])
        for j in range(10):
            train_Data, train_score, test_Data, test_Score = split_train_test(test_X, test_y, i / 100.0)
            l_r.fit(train_Data, train_score)
            loss_10_semp[j] = l_r.loss(test_X, test_y)
        average_loss[i - 10] = loss_10_semp.mean()
        var[i - 10] = loss_10_semp.std()
        y_upper.append(loss_10_semp.mean() + 2 * loss_10_semp.std())
        y_lower.append(loss_10_semp.mean() - 2 * loss_10_semp.std())

    X = list(range(10, 101))
    fig = go.Figure([
        go.Scatter(name='Measurement', x=list(range(10, 100)), y=average_loss, mode='lines',
                   line=dict(color='rgb(0, 255, 255)')),
        go.Scatter(
            x=list(range(10, 100)) + list(range(10, 100))[::-1], y=y_upper + y_lower[::-1], fill='toself',
            fillcolor='rgba(0,153,153,0.2)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False)
    ])
    fig.update_xaxes(title_text="Percentage")
    fig.update_yaxes(title_text="Mean Loss")
    fig.update_layout(title="The Mean Loss as a function of p% with a confidence interval of "
                            "mean(loss)±2 ∗std(loss)")
    fig.show()
