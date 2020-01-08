"""
Creates macro variable features.
"""
from datetime import datetime

import numpy as np
import pandas as pd


def construct_fred(
    src: str
) -> None:
    """
    Main method: Constructs dataset from fred macro dataset.

    Arguments:
        src {str} -- [Directory of sources].
    """
    if not src.endswith("/"):
        src += "/"
    print(f"Reading data from {src}")

    


if __name__ == "__main__":
    construct_fred("./")
