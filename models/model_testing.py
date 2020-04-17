"""
Evaluate model performance on the test set.
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from data_feed import direct_feed

from training_utils import mse, directional_accuracy


def test_rf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: dict
) -> str:
    model = RandomForestRegressor(**config)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_mse = mse(y_test, y_test_pred)
    test_da = directional_accuracy(y_test, y_test_pred)
    report_str = f"Test MSE & Test DA: {test_mse:0.3f} & {test_da * 100:0.3f}%\n"
    return report_str


def test_svr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: dict
) -> str:
    model = SVR(**config)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_mse = mse(y_test, y_test_pred)
    test_da = directional_accuracy(y_test, y_test_pred)
    report_str = f"Test MSE & Test DA: {test_mse:0.3f} & {test_da * 100:0.3f}%\n"
    return report_str


def main():
    # For partial and complete information set.
    # Partial information set.
    svr_alldays_mse = {'tol': 0.01, 'kernel': 'rbf',
                       'gamma': 0.1, 'epsilon': 1, 'C': 1}
    svr_alldays_da = {'tol': 1e-07, 'kernel': 'rbf',
                      'gamma': 0.1, 'epsilon': 1e-06, 'C': 1}

    rf_alldays_mse = {'n_estimators': 54, 'min_samples_split': 2,
                      'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10, 'bootstrap': True}

    rf_alldays_da = {'n_estimators': 130, 'min_samples_split': 10,
                     'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}

    # Complete information set.
    # svr_alldays_mse = {"tol": 1, "kernel": "rbf", "gamma": 1e-06, "epsilon": 0.001, "C": 1e-09}
    # svr_alldays_da = {"tol": 1e-05, "kernel": "rbf", "gamma": 1e-07, "epsilon": 1e-07, "C": 10}

    # rf_alldays_mse = {"n_estimators": 96, "min_samples_split": 10, 
    #                   "min_samples_leaf": 2, "max_features": "log2", "max_depth": 10, "bootstrap": True}

    # rf_alldays_da = {"n_estimators": 41, "min_samples_split": 5, "min_samples_leaf": 4,
    #                  "max_features": None, "max_depth": 14, "bootstrap": False}

    # Experiments on day-of-the-week effect.
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

    # Complete Information.
    # data_src = "../data/ready_to_use/complete_feature_target.csv"
    # Partial Information.
    data_src = "../data/ready_to_use/partial_feature_target.csv"

    with open("./model_testing_log.txt", "w") as f:
        for day, (rf_configs, svr_configs) in zip(day_lst, config_lst):
            f.write(f"================= {day} =================\n\n")
            X_train, X_test, y_train, y_test = direct_feed(
                src=data_src,
                test_start=pd.to_datetime("2019-01-01\n"),
                day=day,
                return_array=True
            )
            f.write(f"X_train @ {X_train.shape}\n\n")
            (f"y_train @ {y_train.shape}\n\n")
            f.write(f"X_test @ {X_test.shape}\n\n")
            f.write(f"y_test @ {y_test.shape}\n\n")

            f.write("======== Random Forest ========\n")
            f.write("RF MSE\n")
            f.write(str(rf_configs[0]) + "\n")
            s = test_rf(X_train, X_test, y_train, y_test, rf_configs[0])
            f.write(s)

            f.write("RF DA\n")
            f.write(str(rf_configs[1]) + "\n")
            s = test_rf(X_train, X_test, y_train, y_test, rf_configs[1])
            f.write(s)

            f.write("======== Support Vector Regressions ========\n")
            f.write("SVR MSE\n")
            f.write(str(svr_configs[0]) + "\n")
            s = test_svr(X_train, X_test, y_train, y_test, svr_configs[0])
            f.write(s)

            f.write("SVR DA\n")
            f.write(str(svr_configs[1]) + "\n")
            s = test_svr(X_train, X_test, y_train, y_test, svr_configs[1])
            f.write(s)


if __name__ == "__main__":
    main()
