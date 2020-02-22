import os
import random

import itertools
import numpy as np
import pandas as pd

from datetime import datetime

import statsmodels.api as sm

from tqdm import tqdm


MASTER_DIR = "./"
LOG_FILE = "./arima_results.csv"

df = pd.read_csv(
    MASTER_DIR + "/data/ready_to_use/returns_norm.csv",
    date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
    index_col=0
)

p = q = range(5)
d = range(2)
pdq_config = list(itertools.product(p, d, q))
seasonal_pdq_config = [
    (x[0], x[1], x[2], 5)
    for x in itertools.product(
        range(2),
        d,
        range(2))
]

random.shuffle(pdq_config)
random.shuffle(seasonal_pdq_config)

print(f"Total: {len(pdq_config) * len(seasonal_pdq_config)} configurations.")

assert os.path.isfile(LOG_FILE)

candidates = list()
with open(LOG_FILE, "w") as log:
    log.write("p,d,q,sp,sd,sq,s,aic\n")
    for param in tqdm(pdq_config):
        for param_seasonal in tqdm(seasonal_pdq_config):
            model = sm.tsa.statespace.SARIMAX(
                df,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            results = model.fit()
            candidates.append({
                "pdq": param,
                "seasonal_pdq": param_seasonal,
                "aic": results.aic
            })
            log.write(
                ",".join(str(x) for x in param + param_seasonal) + "," + str(results.aic) + "\n"
            )
