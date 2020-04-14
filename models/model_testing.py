"""
Evaluate model performance on the test set.
"""
import numpy as np
import pandas as pd

from pprint import pprint

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from data_feed import direct_feed


def test_rf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: dict
) -> None:
    model = RandomForestRegressor(**config)
    model.fit(X_train)
    model.predict(X_test)


def test_svr(X_test: np.ndarray, y_test: np.ndarray, config: dict):
    pass


def main():
    svr_alldays_mse = {}
    svr_alldays_da = {}
    rf_allday_mse = {}
    rf_allday_da = {}

    svr_monday_mse = {}
    svr_monday_da = {}
    rf_monday_mse = {}
    rf_monday_da = {}

    svr_otherdays_mse = {}
    svr_otherdays_da = {}
    rf_otherdays_mse = {}
    rf_otherdays_da = {}

    day_lst = [["Monday"], ["Tuesday", "Wednesday", "Thursday", "Friday"], None]
    config_lst = [
        (rf_monday, svr_monday),
        (rf_otherdays, svr_otherdays),
        (rf_alldays, svr_alldays),
    ]
    for day, (rf_config, svr_config) in zip(day_lst, config_lst):
        print(f"================= {day} =================")
        X_train, X_test, y_train, y_test = direct_feed(
            src="../data/ready_to_use/feature_target_2020-04-05-14:13:42.csv",
            test_start=pd.to_datetime("2019-01-01"),
            day=day,
            return_array=True
        )
        print(f"X_train @ {X_train.shape}")
        print(f"y_train @ {y_train.shape}")
        print(f"X_test @ {X_test.shape}")
        print(f"y_test @ {y_test.shape}")

        print("==== Random Forest ====")
        pprint(rf_config)
        test_rf(X_train, X_test, y_train, y_test, rf_config)

        print("==== Support Vector Regressions ====")
        pprint(svr_config)
        test_svr(X_train, X_test, y_train, y_test, svr_config)


if __name__ == "__main__":
    main()
