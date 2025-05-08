import typing as t
from datetime import datetime

import numpy as np
from nptyping import NDArray

from .linear_cca import LinearCCA  # noqa
from .parser import *  # noqa


def get_now() -> str:
    return datetime.now().strftime("%Y_%m%d_%H_%M_%S")


def batch_generator(
    data1: NDArray, data2: NDArray, batch_size: int, isShuffle: bool = True
) -> t.Iterator[t.Tuple[NDArray, NDArray]]:
    indices = np.arange(len(data1))
    batch = []
    while True:
        if isShuffle:
            np.random.shuffle(indices)
        for index in indices:
            batch.append(index)
            if len(batch) == batch_size:
                yield (data1[batch], data2[batch])
                batch = []
        if len(batch) != 0:
            yield (data1[batch], data2[batch])
            batch = []
