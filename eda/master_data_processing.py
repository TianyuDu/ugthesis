"""
Aggregate data processing utility, generate the ready to use dataset.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def rpna(config: dict):
    raise NotImplementedError


def crude_oil(config: dict):
    raise NotImplementedError


def main(
    src: str,
    config: dict
) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    # Add configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", default="./master_dataset.csv", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config)
    print(f"Read configuration from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)
        print(config)
