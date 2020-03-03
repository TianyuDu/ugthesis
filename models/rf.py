"""
Random Forest
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


from training_utils import directional_accuracy


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


def construct_model(
    config: dict
) -> "RandomForestRegressor":
    model = RandomForestRegressor(**config)
    return model


if __name__ == "__main__":
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
    bootstrap = [True, False]

    # ==== Smaller Profile ====
    # n_estimators = [10]
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [10]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1]
    # # Method of selecting samples for training each tree
    # bootstrap = [True]

    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    DATA_PATH = "../sentiment_data/QA_LMD_data_all_returns.csv"

    model = RandomForestRegressor()
    rf_random = RandomizedSearchCV(
        estimator=model, param_distributions=random_grid,
        n_iter=100,
        scoring={
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'acc': make_scorer(directional_accuracy)
        },
        cv=5, verbose=2, random_state=42, n_jobs=-1,
        return_train_score=True,
        refit=False
    )
    X, y = data_feed(DATA_PATH)
    rf_random.fit(X, y)
    print("======== Best Parameter ========")
    # print(rf_random.best_params_)
    pd.DataFrame.from_dict(rf_random.cv_results_).to_csv("../model_selection_results/rf_results.csv")
