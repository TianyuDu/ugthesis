"""
Creates macro variable features.
"""
from datetime import datetime

import numpy as np
import pandas as pd


FILE_FREQ = {
    "Daily": "D",
    "Monthly": "M",
    "Quarterly": "Q",
    "Weekly_Ending_Monday": "W-MON"
}


def align_dataset(
    src: str,
    start: str = "2000-01-01",
    end: str = "2019-09-30",
    keep_nan: bool = True,
    output_freq: str = "D"
) -> None:
    """
    Main method: Constructs macro indicators (independent variable) from fred dataset.

    Arguments:
        src {str} -- [Directory of sources.]

        start {str} -- [start date of the panel data.]

        end {str} -- [end date of the panel data.]

        keep_nan {bool} -- [whether to drop the entire row (date) if
            missing values encountered in some series]

        output_freq {str} -- [the frequency of constructed dataset, default: daily (D),
            use the standard notation in datetime.]
    """
    if not src.endswith("/"):
        src += "/"
    print(f"Reading data from {src}")
    parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    (start, end) = map(parser, (start, end))

    for (name, freq) in FILE_FREQ.items():
        # TODO: stopped here. align dataset


if __name__ == "__main__":
    align_dataset("./Crude_Oil_Macro_Indicators")
