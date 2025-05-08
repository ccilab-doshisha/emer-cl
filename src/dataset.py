import os
import random
import typing as t

import numpy as np
from nptyping import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

base_path = os.path.join(os.environ["WORKDIR"], "src", "data")
split_seed = random.randint(1, int(1e5))


def load_vggish(dataset_type: str) -> t.List[NDArray]:

    data: NDArray[t.Any, t.Any, 128] = np.load(
        os.path.join(base_path, dataset_type, "vggish_feat.npy"),
        allow_pickle=True,
    )

    if dataset_type == "DEAM":  # for MLP
        data_2d: NDArray[t.Any, 128] = data.mean(axis=1)
        return _split_data(dataset_type, data_2d)

    return _split_data(dataset_type, data)


def load_va(dataset_type: str) -> t.List[NDArray]:

    data: NDArray[t.Any, t.Any, 2] = np.load(
        os.path.join(base_path, dataset_type, "va_data.npy"),
        allow_pickle=True,
    )

    return _split_data(dataset_type, data)


# ========== helper functions for loading ==========


def _split_data(dataset_type: str, data: NDArray) -> t.List[NDArray]:

    if dataset_type == "PMEmo":
        data = pad_sequences(data, padding="post", dtype="float32", value=1e-5)

    train, test = train_test_split(
        data, train_size=0.8, shuffle=True, random_state=split_seed
    )

    return [train, test]


def save_split_seed(save_dir: str) -> None:
    save_path = os.path.join(save_dir, "split_seed.txt")
    with open(save_path, mode="w") as f:
        f.write(str(split_seed))


def load_split_seed(load_dir: str, seed: t.Optional[int] = None) -> None:
    global split_seed
    split_seed = get_seed(load_dir) if seed is None else seed


def get_seed(load_dir: str) -> int:
    path = os.path.join(load_dir, "split_seed.txt")
    with open(path, mode="r") as f:
        seed = f.readline()
    return int(seed)
