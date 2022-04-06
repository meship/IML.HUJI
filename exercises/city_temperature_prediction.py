import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    def to_day_of_year(date):
        return pd.to_datetime(date).day_of_year
    full_data = pd.read_csv(filename, parse_dates=['Date'], date_parser=pd.to_datetime,
                            dayfirst=True).drop_duplicates().dropna()
    full_data = full_data.drop(full_data[(full_data["Temp"] < -1*25)].index)

    features = full_data[["Country", "Day", "Month", "Year"]]
    features["DayOfYear"] = full_data["Date"].apply(to_day_of_year)
    return features, full_data["Temp"]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data, temp = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data["Country"] == "Israel"]
    israel_temp = temp[data[data["Country"] == "Israel"].index]
    fig = px.scatter(israel_data, x="DayOfYear", y=israel_temp)
    fig.show()
    func = lambda t: np.std(israel_temp[data[data["Month"] == t].index])
    israel_data_m = israel_data.groupby('Month')
    i = 9


    # Question 3 - Exploring differences between countries
    #raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    #raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    #raise NotImplementedError()