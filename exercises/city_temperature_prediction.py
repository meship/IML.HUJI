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
    data_with_temp = features = pd.concat([data, temp], axis=1, join='inner')
    israel_data = data_with_temp[data_with_temp["Country"] == "Israel"]
    fig = px.scatter(israel_data, x="DayOfYear", y="Temp", color=israel_data['Year'].astype(str),
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     title="Temperature as a Function of Day of Year in Israel")
    fig.update_traces(marker_size=10)
    fig.show()
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Month = pd.DataFrame({'Month': months})
    israel_data_m = israel_data.groupby('Month').agg({'Temp': 'std'})
    israel_data_m = pd.concat([israel_data_m, Month], axis=1, join='inner')
    fig = px.bar(israel_data_m, x='Month', y='Temp', color_discrete_sequence=px.colors.qualitative.Pastel,
                 title="Temperature STD over the Months in Israel")
    fig.show()



    # Question 3 - Exploring differences between countries
    temp_by_mean_std = data_with_temp.groupby(['Month', 'Country']).agg({'Temp': ['mean', 'std']})
    temp_by_mean_std.columns = ['Mean Temp', 'STD Temp']
    temp_by_mean_std = temp_by_mean_std.reset_index()
    fig = px.line(temp_by_mean_std, x='Month', y='Mean Temp', error_y='STD Temp', color='Country',
                  color_discrete_sequence=px.colors.qualitative.Vivid,
                  title="The Mean Temperature With STD Error, in Each Month")
    fig.show()
    # Question 4 - Fitting model for different values of `k`
    relevant_data = israel_data[['DayOfYear']]
    train_X, train_y, test_X, test_y = split_train_test(relevant_data, israel_data['Temp'])
    poly_lost = []
    for i in range(1,11):
        p_r = PolynomialFitting(i)
        p_r.fit(train_X['DayOfYear'], train_y)
        poly_lost.append(p_r.loss(test_X['DayOfYear'], test_y))
    loss_as_k = pd.DataFrame({'power': list(range(1,11)), 'loss': poly_lost})
    print("the losses are:")
    print(loss_as_k)
    fig = px.bar(loss_as_k, x='power', y='loss',
                 title='The Loss of the Polynomial Fitting Model According  '
                       'to the Polynomial Rank')
    fig.show()


    # Question 5 - Evaluating fitted model on different countries
    p_r = PolynomialFitting(5)
    p_r.fit(relevant_data['DayOfYear'], israel_data['Temp'])
    loss = []
    jordan_data = data_with_temp[data_with_temp["Country"] == "Jordan"]
    loss.append(p_r.loss(jordan_data['DayOfYear'], jordan_data['Temp']))
    SA_data = data_with_temp[data_with_temp["Country"] == "South Africa"]
    loss.append(p_r.loss(SA_data['DayOfYear'], SA_data['Temp']))
    NL_data = data_with_temp[data_with_temp["Country"] == "The Netherlands"]
    loss.append(p_r.loss(NL_data['DayOfYear'], NL_data['Temp']))
    fig = px.bar(x=['Jordan',"South Africa","The Netherlands"], y=loss,
                 color_discrete_sequence=px.colors.qualitative.Set2, labels={'x': 'Country', 'y':'MSE'},
                 title='The Model’s Error Over Each of the Other Countries According to '
                       'the Fitted Model on Israel’s Data')
    fig.show()