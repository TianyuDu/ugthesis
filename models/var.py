"""
Vector Autoregression and other time series methods.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

from data_feed import direct_feed

data_src = "../data/ready_to_use/feature_target_2020-04-05-14:13:42.csv"


def arima(
    return_train: pd.DataFrame,
    return_test: pd.DataFrame
) -> None:
    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())


def main() -> None:
    # ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    X_train, X_test, y_train, y_test = direct_feed(
        src=data_src,
        test_start=pd.to_datetime("2019-01-01"),
        day=None,
        return_array=False
    )
    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_test @ {X_test.shape}")
    print(f"y_test @ {y_test.shape}")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
