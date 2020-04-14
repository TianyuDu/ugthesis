"""
Evaluate model performance on the test set.
"""
import numpy as np
import pandas as pd

from data_feed import direct_feed


def test_rf(X_test: np.ndarray, y_test: np.ndarray, config: dict):
    pass


def test_svr(X_test: np.ndarray, y_test: np.ndarray, config: dict):
    pass


def main():
    for day in [["Monday"], ["Tuesday", "Wednesday", "Thursday", "Friday"], None]:
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
        test_rf(X_test, y_test)
        test_svr(X_test, y_test)


if __name__ == "__main__":
    main()
