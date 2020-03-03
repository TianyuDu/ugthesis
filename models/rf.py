"""
Random Forest
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


from training_utils import directional_accuracy

from typing import Union

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import data_feed


def construct_model(
    config: dict
) -> "RandomForestRegressor":
    model = RandomForestRegressor(**config)
    return model


def main(
    result_path: Union[str, None] = None
) -> None:
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # ==== Smaller Profile For Testing Purpose====
    n_estimators = [10]
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [10]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # ================================================

    # Create the random grid
    grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    model = RandomForestRegressor()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring={
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'acc': make_scorer(directional_accuracy)
        },
        cv=5,
        verbose=2,
        n_jobs=-1,
        return_train_score=True,
        refit=False
    )
    X_train, y_train, X_test, y_test = data_feed.regression_feed()
    # Flatten
    N_train, L, D = X_train.shape
    N_test = X_test.shape[0]

    X_train = X_train.reshape(N_train, -1)
    y_train = y_train.reshape(N_train,)

    X_test = X_test.reshape(N_test, -1)
    y_test = y_test.reshape(N_test,)

    grid_search.fit(X_train, y_train)
    # print("======== Best Parameter ========")
    # print(grid_search.best_params_)
    if result_path is None:
        pd.DataFrame.from_dict(grid_search.cv_results_).to_csv(
            "../model_selection_results/rf_results.csv")
    else:
        pd.DataFrame.from_dict(grid_search.cv_results_).to_csv(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=str)
    args = parser.parse_args()
    main(args.report_dir)
