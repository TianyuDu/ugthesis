"""
Support Vector Regressor.
"""
import argparse
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.svm import SVR

from data_feed import direct_feed
from training_utils import directional_accuracy, mape


def construct_model(
    config: dict
) -> "SVR":
    model = SVR(**config)
    return model


def main(
    result_path: str,
    n_iter: int = 1000
) -> None:
    # Create the random grid
    random_grid = {
        "kernel": ["rbf"],
        "gamma": [10**x for x in range(-10, 2)],
        "tol": [10**x for x in range(-10, 2)],
        "epsilon": [10**x for x in range(-10, 2)],
        "C": [10**x for x in range(-10, 2)]
    }

    model = SVR()
    grid_search = RandomizedSearchCV(
        estimator=model, param_distributions=random_grid,
        n_iter=n_iter,
        scoring={
            "neg_mse": "neg_mean_squared_error",
            "dir_acc": make_scorer(directional_accuracy),
            "mape": make_scorer(mape)
        },
        cv=5, verbose=10, random_state=42, n_jobs=-1,
        return_train_score=True,
        refit=False
    )

    # Complete Information.
    # data_src = "../data/ready_to_use/complete_feature_target.csv"
    # Partial Information.
    data_src = "../data/ready_to_use/partial_feature_target.csv"

    # Datafeed: avaiable options:
    # ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    X_train, X_test, y_train, y_test = direct_feed(
        src=data_src,
        test_start=pd.to_datetime("2019-01-01"),
        day=None,
        return_array=True
    )
    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_test @ {X_test.shape}")
    print(f"y_test @ {y_test.shape}")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # Flatten
    # N_train, L, D = X_train.shape
    # N_test = X_test.shape[0]

    # X_train = X_train.reshape(N_train, -1)
    # y_train = y_train.reshape(N_train,)

    # X_test = X_test.reshape(N_test, -1)
    # y_test = y_test.reshape(N_test,)

    grid_search.fit(X_train, y_train)
    # print("======== Best Parameter ========")
    report = pd.DataFrame.from_dict(grid_search.cv_results_)
    report.sort_values(by=["mean_test_dir_acc"], ascending=False, inplace=True)
    # Move the dir acc column to the first column
    columns = ["mean_test_dir_acc", "mean_test_mape"] + \
        [col for col in report.columns if col != "mean_test_dir_acc"]
    report = report[columns]
    if result_path is None:
        report.to_csv(
            f"../model_selection_results/svr_cv_results_{n_iter}_iters.csv")
    else:
        report.to_csv(result_path)
    print(f"Data source: {data_src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--n_iter", type=int, default=100)
    args = parser.parse_args()
    start_time = datetime.now()
    main(args.log_dir, args.n_iter)
    print(
        f"Time taken for {args.n_iter} samples: {datetime.now() - start_time}")
