import os
import random
import typing as t
from pdb import set_trace as st

import numpy as np
from nptyping import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

base_path = os.path.join(os.environ["WORKDIR"], "src", "data")
split_seed = random.randint(1, int(1e5))


def load_vggish_10_folds(dataset_type: str, fold_idx: int) -> t.List[NDArray]:

    trainData: NDArray[t.Any, t.Any, 128] = np.load(
            os.path.join(base_path, dataset_type, "10_fold_cv_dataset", str(fold_idx).zfill(2), "trainData.npy"),
        allow_pickle=True,
    )

    testData: NDArray[t.Any, t.Any, 128] = np.load(
            os.path.join(base_path, dataset_type, "10_fold_cv_dataset", str(fold_idx).zfill(2), "testData.npy"),
        allow_pickle=True,
    )

    if dataset_type == "DEAM":  # for MLP
        trainData_2d: NDArray[t.Any, 128] = trainData.mean(axis=1)
        testData_2d: NDArray[t.Any, 128] = testData.mean(axis=1)
        # return _split_data(dataset_type, data_2d)
        return [trainData_2d, testData_2d]

    elif dataset_type == "PMEmo":
        trainData = pad_sequences(trainData, padding="post", dtype="float32", value=1e-5)
        testData = pad_sequences(testData, padding="post", dtype="float32", value=1e-5)
        return [trainData, testData]

    else:
        print("Incorrect input dataset name %s!" % dataset_type)
        return []
    
    #return _split_data(dataset_type, data)


def load_va_10_folds(dataset_type: str, fold_idx: int) -> t.List[NDArray]:

    trainData: NDArray[t.Any, t.Any, 2] = np.load(
        os.path.join(base_path, dataset_type, "10_fold_cv_dataset", str(fold_idx).zfill(2), "trainLabels.npy"),
        allow_pickle=True,
    )

    testData: NDArray[t.Any, t.Any, 2] = np.load(
        os.path.join(base_path, dataset_type, "10_fold_cv_dataset", str(fold_idx).zfill(2), "testLabels.npy"),
        allow_pickle=True,
    )

    if dataset_type == "PMEmo":
        trainData = pad_sequences(trainData, padding="post", dtype="float32", value=1e-5)
        testData = pad_sequences(testData, padding="post", dtype="float32", value=1e-5)

    #return _split_data(dataset_type, data)
    return [trainData, testData]


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
