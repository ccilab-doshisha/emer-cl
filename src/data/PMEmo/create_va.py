import os
import typing as t

import numpy as np
import pandas as pd
from nptyping import NDArray

BASE: str = os.path.join(os.environ["WORKDIR"], "src", "data", "PMEmo")
DATASET: str = os.path.join(BASE, "dataset", "PMEmo2019")
EMOTION: str = os.path.join(DATASET, "annotations")


def numptToList(arr: NDArray) -> t.List[t.Any]:
    l: t.List[t.Any] = arr.tolist()
    return l


def listToNumpy(arr: t.List[t.Any]) -> NDArray:
    import warnings

    warnings.filterwarnings("error")
    try:
        return np.asarray(arr)
    # VisibleDeprecationWarning
    except Warning:
        return np.asarray(arr, dtype=object)


def save() -> None:

    if os.path.isfile(os.path.join(BASE, "va_data.npy")):
        return
    df = pd.read_csv(os.path.join(EMOTION, "dynamic_annotations.csv"))

    va_data: t.List[t.Any] = []
    va_append = va_data.append

    remain_music_indices = np.load(
        os.path.join(BASE, "remain_music_indices.npy")
    )
    df_gb = df.groupby("musicId")

    for i, idx in enumerate(remain_music_indices):
        va: NDArray = (
            df_gb.get_group(idx)
            .loc[:, ["Valence(mean)", "Arousal(mean)"]]
            .values
        )
        va_append(numptToList(va))
        if (i % 100 == 0) and (not (i == 0)):
            print("\033[1;32m%s\033[0m %s" % ("[LOG]", "#%d is finished" % (i)))

    va_data = listToNumpy(va_data)
    np.save(os.path.join(BASE, "va_data"), va_data)


def main() -> None:
    save()


if __name__ == "__main__":
    main()
