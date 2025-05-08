# reorganise metadata

import os
import typing as t

import numpy as np
import pandas as pd

BASE: str = os.path.join(os.environ["WORKDIR"], "src", "data", "PMEmo")
DATASET: str = os.path.join(BASE, "dataset", "PMEmo2019")
METADATA: str = os.path.join(DATASET, "metadata")


def main() -> None:

    if os.path.isfile(os.path.join(METADATA, "metadata.csv")):
        return

    if not os.path.exists(METADATA):
        os.makedirs(METADATA)

    orig_matadata_path = os.path.join(DATASET, "metadata_discard.csv")
    orig_matadata = pd.read_csv(orig_matadata_path)
    data = orig_matadata[["musicId", "fileName", "artist", "title"]]
    data_num = len(data)
    data = np.hstack((data, np.repeat("pop", data_num).reshape((data_num, 1))))

    headers: t.List[str] = ["musicId", "fileName", "artist", "title", "genre"]
    df_out = pd.DataFrame(data=data, columns=headers).set_index("musicId")
    df_out.to_csv(os.path.join(METADATA, "metadata.csv"))


if __name__ == "__main__":
    main()
