import os
import typing as t

import numpy as np
import pandas as pd
from nptyping import NDArray

BASE: str = os.path.join(os.environ["WORKDIR"], "src", "data", "DEAM")
DATASET: str = os.path.join(BASE, "dataset")
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

    va_data: t.List[t.Any] = []
    va_append = va_data.append

    # include arousal and valence
    orig_dir = os.path.join(
        EMOTION,
        "annotations per each rater",
        "dynamic (per second annotations)",
    )

    # It corresponds to the file number
    indices = sorted(
        [
            int(f.name.split(".")[0])
            for f in os.scandir(os.path.join(orig_dir, "arousal"))
            if f.is_file()
        ]
    )

    for i, idx in enumerate(indices):

        a_df = pd.read_csv(os.path.join(orig_dir, "arousal", "%d.csv" % idx))
        v_df = pd.read_csv(os.path.join(orig_dir, "valence", "%d.csv" % idx))

        if a_df.columns.str.startswith("WorkerId")[0]:
            a_df = a_df.iloc[:, ~a_df.columns.str.startswith("WorkerId")]
            v_df = v_df.iloc[:, ~v_df.columns.str.startswith("WorkerId")]

        # average of valence/arousal seq for subject
        a_data = numptToList(a_df.values.mean(axis=0))
        v_data = numptToList(v_df.values.mean(axis=0))

        # only use 60 frame
        if len(a_data) != 60:
            a_data = a_data[:60]
            v_data = a_data[:60]

        """
        ex.
        valence_seq = [1, 2, 3]
        arousal_seq = [6, 7, 8]
        va_tmp = [[1, 2], [2, 7], [3, 8]]
        """
        va_tmp = np.stack([v_data, a_data], -1)
        va_append(numptToList(va_tmp))

        if (i % 100 == 0) and (not (i == 0)):
            print("\033[1;32m%s\033[0m%s" % ("[LOG] ", "#%d is finished" % (i)))

    va_data = listToNumpy(va_data)
    np.save(os.path.join(BASE, "va_data"), va_data)


def main() -> None:
    save()


if __name__ == "__main__":
    main()
