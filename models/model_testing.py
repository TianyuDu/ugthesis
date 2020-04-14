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
    svr_alldays_mse = {"tol": 1, "kernel": "rbf", "gamma": 1e-06, "epsilon": 0.001, "C": 1e-09}
    svr_alldays_da = {"tol": 1e-05, "kernel": "rbf", "gamma": 1e-07, "epsilon": 1e-07, "C": 10}

    rf_alldays_mse = {"n_estimators": 96, "min_samples_split": 10,
                     "min_samples_leaf": 2, "max_features": "log2", "max_depth": 10, "bootstrap": True}

    rf_alldays_da = {"n_estimators": 41, "min_samples_split": 5, "min_samples_leaf": 4,
                    "max_features": None, "max_depth": 14, "bootstrap": False}

    svr_monday_mse = {"tol": 0.001, "kernel": "rbf", "gamma": 1e-10, "epsilon": 0.0001, "C": 10}
    svr_monday_da = {"tol": 0.1, "kernel": "rbf", "gamma": 1e-06, "epsilon": 1e-06, "C": 1}

    rf_monday_mse = {"n_estimators": 115, "min_samples_split": 10,
                     "min_samples_leaf": 4, "max_features": "log2", "max_depth": 38, "bootstrap": True}

    rf_monday_da = {"n_estimators": 116, "min_samples_split": 2, "min_samples_leaf": 1,
                    "max_features": "log2", "max_depth": 38, "bootstrap": False}

    svr_otherdays_mse = {"tol": 0.1, "kernel": "rbf", "gamma": 0.1, "epsilon": 0.0001, "C": 10}
    svr_otherdays_da = {"tol": 0.1, "kernel": "rbf", "gamma": 1e-09, "epsilon": 1e-07, "C": 1}

    rf_otherdays_mse = {"n_estimators": 88, "min_samples_split": 5,
                        "min_samples_leaf": 4, "max_features": "log2", "max_depth": 10, "bootstrap": True}

    rf_otherdays_da = {"n_estimators": 157, "min_samples_split": 2,
                       "min_samples_leaf": 1, "max_features": None, "max_depth": 10, "bootstrap": False}

    day_lst = [["Monday"], ["Tuesday", "Wednesday", "Thursday", "Friday"], None]

    config_lst = [
        ([rf_monday_mse, rf_monday_da], [svr_monday_mse, svr_monday_da]),
        ([rf_otherdays_mse, rf_otherdays_da], [svr_otherdays_mse, svr_otherdays_da]),
        ([rf_alldays_mse, rf_alldays_da], [svr_alldays_mse, svr_alldays_da]),
    ]
    for day, (rf_configs, svr_configs) in zip(day_lst, config_lst):
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

        print("======== Random Forest ========")
        print("RF MSE")
        pprint(rf_configs[0])
        test_rf(X_train, X_test, y_train, y_test, rf_configs[0])

        print("RF DA")
        pprint(rf_configs[1])
        test_rf(X_train, X_test, y_train, y_test, rf_configs[1])

        print("======== Support Vector Regressions ========")
        print("SVR MSE")
        pprint(svr_configs[0])
        test_svr(X_train, X_test, y_train, y_test, svr_configs[0])

        print("SVR DA")
        pprint(rf_configs[1])
        pprint(svr_configs)
        test_svr(X_train, X_test, y_train, y_test, svr_configs[1])


if __name__ == "__main__":
    main()
