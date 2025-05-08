# reorganise metadata

import csv
import os
import shutil
import typing as t

import pandas as pd
from tqdm import tqdm

BASE_: str = os.path.join(os.environ["WORKDIR"], "src", "data")
BASE: str = os.path.join(BASE_, "DEAM")
DATASET: str = os.path.join(BASE, "dataset")
METADATA: str = os.path.join(DATASET, "metadata")
AUDIO: str = os.path.join(DATASET, "MEMD_audio")


def deleteTabChar(lines: t.List[t.List[str]]) -> t.List[t.List[str]]:
    _lines: t.List[t.List[str]] = []
    for line in lines:
        _line: t.List[str] = []
        for element in line:
            if type(element) is str:
                element = element.replace("\t", "")
            _line.append(element)
        _lines.append(_line)
    return _lines


def metadata_clipper(path: str) -> None:
    """
    Each metadata has a different data format header,
    so only the necessary information is extracted.
    The original metadata will be copied into "original" directory,
    and then the original metadata file will be overwritten.
    """

    # create_original_dir
    original_dir: str = os.path.join(METADATA, "original")
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
    # copy original file into original dir
    shutil.copy(path, original_dir)

    # load_csv
    csv_list: t.List[t.List[str]] = []
    with open(path) as f:
        reader = csv.reader(f)
        for line in reader:
            csv_list.append(line)

    # Delete empty characters
    header: t.List[str] = [value for value in csv_list[0] if value != ""]
    csv_list = deleteTabChar([line[: len(header)] for line in csv_list])

    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)


def main() -> None:

    if os.path.isfile(os.path.join(METADATA, "metadata.csv")):
        return

    file_nums: t.List[int] = sorted(
        [int(f.name.split(".")[0]) for f in os.scandir(AUDIO) if f.is_file()]
    )

    # metadata_2013 <= 1000
    meta2013_path = os.path.join(METADATA, "metadata_2013.csv")
    metadata_clipper(meta2013_path)
    meta2013 = pd.read_csv(meta2013_path)
    meta2013_musicId = meta2013["song_id"].values.tolist()
    # 1000 < metadata_2014 <= 2000
    meta2014_path = os.path.join(METADATA, "metadata_2014.csv")
    metadata_clipper(meta2014_path)
    meta2014 = pd.read_csv(meta2014_path)
    meta2014_musicId = meta2014["Id"].values.tolist()
    # 2000 < metadata_2015 <= else (<=2058)
    meta2015_path = os.path.join(METADATA, "metadata_2015.csv")
    metadata_clipper(meta2015_path)
    meta2015 = pd.read_csv(meta2015_path)
    meta2015_musicId = meta2015["id"].values.tolist()

    metadata: t.List[t.Any] = []

    """
    The header names are different, so only the common points are summarised
    """
    for musicId in tqdm(file_nums):
        musicName = "%d.wav" % musicId
        # summarise metadata
        if musicId in meta2013_musicId:
            tmp = meta2013[meta2013["song_id"] == musicId]
            metadata.append(
                [musicId, musicName]
                + tmp[["Artist", "Song title", "Genre"]].values.tolist()[0]
            )
        elif musicId in meta2014_musicId:
            tmp = meta2014[meta2014["Id"] == musicId]
            metadata.append(
                [musicId, musicName]
                + tmp[["Artist", "Track", "Genre"]].values.tolist()[0]
            )
        elif musicId in meta2015_musicId:
            tmp = meta2015[meta2015["id"] == musicId]
            metadata.append(
                [musicId, musicName]
                + tmp[["artist", "title", "genre"]].values.tolist()[0]
            )
        else:
            print("ERROR : miss match")
            exit(1)

    headers: t.List[str] = ["musicId", "fileName", "artist", "title", "genre"]
    df_out = pd.DataFrame(metadata, columns=headers).set_index("musicId")
    df_out.to_csv(os.path.join(METADATA, "metadata.csv"))


if __name__ == "__main__":
    main()
