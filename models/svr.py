"""
Baseline Support Vector Regressor.
"""
import numpy as np
import pandas as pd

from grid_search_util import profile_generator, grid_search
from training_utils import data_feed, training_pipeline, directional_accuracy


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from sklearn.svm import SVR


def construct_model(
    config: dict
) -> "SVR":
    model = SVR(**config)
    return model


if __name__ == "__main__":
    # Create the random grid
    random_grid = {
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
        "tol": [10**x for x in range(-10, 0)],
        "epsilon": [10**x for x in range(-10, 0)]
    }

    DATA_PATH = "../sentiment_data/QA_LMD_data_all_returns.csv"

    model = SVR()
    cv = RandomizedSearchCV(
        estimator=model, param_distributions=random_grid,
        n_iter=500,
        scoring={
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'acc': make_scorer(directional_accuracy)
        },
        cv=5, verbose=2, random_state=42, n_jobs=-1,
        return_train_score=True,
        refit=False
    )
    X, y = data_feed(DATA_PATH)
    cv.fit(X, y)
    print("======== Best Parameter ========")
    # print(rf_random.best_params_)
    pd.DataFrame.from_dict(cv.cv_results_).to_csv(
        "../model_selection_results/svr_results.csv")
