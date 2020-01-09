"""
Creates macro variable features.
"""
import argparse

from datetime import datetime

import numpy as np
import pandas as pd


FILE_FREQ = {
    "Crude_Oil_Macro_Indicators_Daily": "D",
    "Crude_Oil_Macro_Indicators_Monthly": "MS",
    "Crude_Oil_Macro_Indicators_Quarterly": "QS",
    "Crude_Oil_Macro_Indicators_Weekly_Ending_Monday": "W-MON"
}


def align_dataset(
    src: str,
    start: str = "2000-01-01",
    end: str = "2019-09-30",
    keep_nan: bool = True,
    output_freq: str = "D"
) -> pd.DataFrame:
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

    data_collection = list()
    for (name, freq) in FILE_FREQ.items():
        print(f"Convert {name} to {freq} frequency.")
        df = pd.read_csv(
            f"{src}{name}.txt",
            index_col=0,
            header=0,
            sep="\t",
            parse_dates=["DATE"],
            date_parser=parser
        )
        df.replace(".", np.nan, inplace=True)
        # df = df.astype(np.float32)
        print(f"Missing values for {name}: \n{np.mean(df.isna(), axis=0)}")
        df = df.asfreq(freq)
        # Create the features in previous time step.
        prev_colns = ["Prev_" + c for c in df.columns]
        df_prev = df.shift(1)
        df_prev.columns = prev_colns
        df_all = pd.concat([df, df_prev], axis=1)
        data_collection.append(df_all.asfreq("D").ffill()[prev_colns].copy())
    panel = pd.concat(data_collection, axis=1)
    # Subsetting.
    panel = panel[start: end]
    # Forward filling.
    panel = panel.astype(np.float32)
    panel.info()
    return panel


if __name__ == "__main__":
    panel = align_dataset("./Crude_Oil_Macro_Indicators")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log_dir", type=str, default="./")
    args = argparser.parse_args()
    log_dir = args.log_dir
    if not log_dir.endswith("/"):
        log_dir += "/"
    print(f"Write file to {log_dir}")
    panel.to_csv(f"{log_dir}panel.csv")
