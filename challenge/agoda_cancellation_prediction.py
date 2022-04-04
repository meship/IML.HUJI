from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def get_canceled_days_before_checkout(canceled_date):
    if pd.isnull(canceled_date):
        return 0
    return 1


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[[
        "hotel_star_rating",
        "charge_option",
        "no_of_room",
        "no_of_extra_bed",
        "original_selling_amount",
        "no_of_children",
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed", "request_twinbeds", "request_airport", "request_earlycheckin"]]

    # features["abroad"] = np.where(full_data["hotel_country_code"] == full_data["origin_country_code"], 0, 1)
    features["request_nonesmoke"] = features["request_nonesmoke"].fillna(0)
    features["request_latecheckin"] = features["request_latecheckin"].fillna(0)
    features["request_highfloor"] = features["request_highfloor"].fillna(0)
    features["request_largebed"] = features["request_largebed"].fillna(0)
    features["request_twinbeds"] = features["request_twinbeds"].fillna(0)
    features["request_airport"] = features["request_airport"].fillna(0)
    features["request_earlycheckin"] = features["request_earlycheckin"].fillna(0)

    features = features.dropna()

    # grades_for_accom = {'Guest House / Bed & Breakfast': 5, 'Holiday Park / Caravan Park': 2, 'Apartment': 3,
    #                     'Bungalow': 1, 'Love Hotel': 4,'Boat / Cruise': 9, 'Inn': 5, 'Lodge': 3,'Homestay':0,
    #                     'Ryokan': 7, 'Resort': 9, 'Private Villa': 5, 'Tent': 1, 'Serviced Apartment': 4,
    #                     'Capsule Hotel': 3, 'Hostel': 5, 'Hotel': 8, 'Resort Villa': 10, 'Motel': 6, 'UNKNOWN': 5,
    #                     'Home': 0}
    # features['accommadation_type_name'].replace(grades_for_accom, inplace=True)

    grades_for_charge = {'Pay Now': 3, 'Pay Later': 2, 'Pay at Check-in': 1}
    features['charge_option'] = features['charge_option'].replace(grades_for_charge)

    labels = full_data['cancellation_datetime'].apply(lambda t: get_canceled_days_before_checkout(t))

    return features, labels


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    y = estimator.predict(X)
    pd.DataFrame(y, columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")

    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X.to_numpy(), train_y.to_numpy())
    #estimator.predict(train_X)

    # Store model predictions over test set
    evaluate_and_export(estimator, train_X.to_numpy(), "try.csv")
