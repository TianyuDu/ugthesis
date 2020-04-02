"""
Baseline Support Vector Regressor.
"""
import argparse
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVR

import data_feed
from training_utils import directional_accuracy


def construct_model(
    config: dict
) -> "SVR":
    model = SVR(**config)
    return model


def main(result_path: str) -> None:
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
        n_iter=5,
        scoring={
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'acc': make_scorer(directional_accuracy)
        },
        cv=5, verbose=10, random_state=42, n_jobs=-1,
        return_train_score=True,
        refit=False
    )

    # Datafeed:
    # ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # X_train, y_train, X_test, y_test = data_feed.feed(
    #     include="master",
    #     day=None,
    #     task="regression"
    # )
    X, y = data_feed.direct_feed("../data/ready_to_use/xyt/")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_test @ {X_test.shape}")
    print(f"y_test @ {y_test.shape}")
    # Flatten
    # N_train, L, D = X_train.shape
    # N_test = X_test.shape[0]

    # X_train = X_train.reshape(N_train, -1)
    # y_train = y_train.reshape(N_train,)

    # X_test = X_test.reshape(N_test, -1)
    # y_test = y_test.reshape(N_test,)

    grid_search.fit(X_train, y_train)
    # print("======== Best Parameter ========")
    # print(grid_search.best_params_)
    if result_path is None:
        pd.DataFrame.from_dict(grid_search.cv_results_).to_csv(
            "../model_selection_results/svr_cv_results.csv")
    else:
        pd.DataFrame.from_dict(grid_search.cv_results_).to_csv(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=str)
    args = parser.parse_args()
    main(args.report_dir)