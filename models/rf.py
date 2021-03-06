"""
Random Forest
"""
import argparse
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from data_feed import direct_feed
from training_utils import directional_accuracy, mape


def construct_model(
    config: dict
) -> "RandomForestRegressor":
    model = RandomForestRegressor(**config)
    return model


def main(
    result_path: Union[str, None] = None,
    n_iter: int = 1000
) -> None:
    # n_estimators = [int(x) for x in np.linspace(start=1, stop=500, num=200)]
    n_estimators = list(range(1, 201))
    max_features = ["auto", "log2", None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=22)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # ==== Smaller Profile For Testing Purpose====
    # n_estimators = [10]
    # max_features = ["auto", "sqrt"]
    # # Maximum number of levels in tree
    # max_depth = [10]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1]
    # # Method of selecting samples for training each tree
    # bootstrap = [True]

    # ================================================

    # Create the random grid
    grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap
    }

    model = RandomForestRegressor()
    grid_search = RandomizedSearchCV(
        estimator=model,
        n_iter=n_iter,
        param_distributions=grid,
        scoring={
            "neg_mse": "neg_mean_squared_error",
            "dir_acc": make_scorer(directional_accuracy),
            "mape": make_scorer(mape)
        },
        cv=5,
        verbose=10,
        n_jobs=-1,
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
            f"../model_selection_results/rf_cv_results_{n_iter}_iters.csv")
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
    print(f"Time taken for {args.n_iter} samples: {datetime.now() - start_time}")
