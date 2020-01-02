"""
Utilities for training neural networks.
"""
from typing import List, Tuple


def batch_sampler(
    batch_size: int,
    data_size: int
) -> List[Tuple[int]]:
    """
    Generates a sequence of (low, high) index pairs for mini-batches.
    Usage:
        for (low, high) in batch_sampler:
            mini_batch = X[low: high, :]
    """
    if batch_size >= data_size:
        return (0, data_size)  # returns the entire dataset.
    lst = list()
    low = 0
    for i in range(data_size // batch_size):
        high = low + batch_size
        lst.append((low, high))
        low = high
    lst.append((low, data_size))
    return lst
