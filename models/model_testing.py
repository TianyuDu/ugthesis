"""
Evaluate model performance on the test set.
"""
import numpy as np
import pandas as pd

from pprint import pprint

from data_feed import direct_feed


def test_rf(X_test: np.ndarray, y_test: np.ndarray, config: dict):
    pass


def test_svr(X_test: np.ndarray, y_test: np.ndarray, config: dict):
    pass


def main():
    day_lst = [["Monday"], ["Tuesday", "Wednesday", "Thursday", "Friday"], None]
    config_lst = [(), (), ()]
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
        test_rf(X_test, y_test, rf_config)

        print("==== Support Vector Regressions ====")
        pprint(svr_config)
        test_svr(X_test, y_test, svr_config)


if __name__ == "__main__":
    main()
