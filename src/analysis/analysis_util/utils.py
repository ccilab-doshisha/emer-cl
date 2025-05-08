import json
import os
import sys
import typing as t

import numpy as np
import pandas as pd
from nptyping import Float, NDArray
from sklearn.model_selection import train_test_split

ROOT: str = os.path.join(os.environ["WORKDIR"], "src")
sys.path.append(ROOT)
from dataset import get_seed, load_split_seed, load_va, load_vggish  # noqa

JSON_PATH: str = os.path.join(ROOT, "analysis", "json")


def _get_seed(load_dir: str) -> int:
    seed: int = get_seed(load_dir)
    load_split_seed(load_dir)
    return seed


def set_split_seed(split_seed: int) -> None:
    load_split_seed(load_dir="", seed=split_seed)


def _split_to_get_test(data: NDArray, seed: int) -> NDArray:
    _, test_sample = train_test_split(
        data, train_size=0.8, shuffle=True, random_state=seed
    )
    return test_sample


def _variable_seq_mean(seq: NDArray) -> Float:
    # (step, dim)
    # the same => lambda x: x.mean(axis=0), seq
    return seq.mean(axis=0)


# load average of emotion
def get_emotion(dataset_name: str) -> NDArray:
    # << NEED TO LOAD SEED >>
    _, test_emotion = load_va(dataset_name)
    if dataset_name == "DEAM":
        return test_emotion.mean(axis=1)
    else:
        # Variable-length sequences
        test_emotion_mean: t.List[Float] = list(
            map(_variable_seq_mean, test_emotion)
        )  # noqa
        return np.asarray(test_emotion_mean)


# load average of vggish
def get_vggish(dataset_name: str) -> NDArray:
    # << NEED TO LOAD SEED >>
    _, test_vggish = load_vggish(dataset_name)
    if dataset_name == "DEAM":
        return test_vggish
    else:
        # Variable-length sequences
        test_vggish_mean: t.List[Float] = list(
            map(_variable_seq_mean, test_vggish)
        )  # noqa
        return np.asarray(test_vggish_mean)


# helper function for metadata
def _get_metadata_orig(metadata_path: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    return metadata.fillna("NAN")


# metadata load (only DEAM)
def get_metadata(dataset_path: str, seed: int) -> NDArray:
    metadata_path = os.path.join(dataset_path, "metadata", "metadata.csv")
    metadata = _get_metadata_orig(metadata_path)
    metadata = metadata[["fileName", "artist", "title", "genre"]]
    metadata = metadata.to_numpy()

    # excerpt test metadata
    test_metadata = _split_to_get_test(metadata, seed)
    return test_metadata


def get_music_id(dataset_path: str, seed: int) -> NDArray:
    metadata_path = os.path.join(dataset_path, "metadata", "metadata.csv")
    metadata = _get_metadata_orig(metadata_path)
    music_id = metadata["musicId"].to_list()

    # excerpt test music_id
    test_music_id = _split_to_get_test(music_id, seed)
    return test_music_id


def get_emotion_classes(dataset_name: str) -> NDArray:
    classes_npy_path = os.path.join(
        os.environ["WORKDIR"],
        "src",
        "analysis",
        "category",
        "npy",
        dataset_name,
        "category_classes.npy",
    )
    category_classes = np.load(classes_npy_path)
    return category_classes


def save_dict_as_json(
    save_dict: t.Dict[str, t.Any], dataset_name: str, filename: str
) -> None:
    save_dir = os.path.join(JSON_PATH, dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(save_dict, f)


def load_json_as_dict(
    dataset_name: str, filename: str
) -> t.Optional[t.Dict[str, t.Any]]:
    path: str = os.path.join(JSON_PATH, dataset_name, filename)
    if not os.path.isfile(path):
        return None

    loaded_dict: t.Dict[str, t.Any]
    with open(path) as f:
        loaded_dict = json.load(f)
    return loaded_dict


"""for check"""


def main() -> None:
    # For DEAM
    # sample_time is need to change to real time stamp saved
    model_timestamp = "sample_time"
    model_path = os.path.join(
        os.environ["WORKDIR"], "src", "model", "DEAM", model_timestamp
    )
    seed: int = _get_seed(model_path)

    analysis_eval_path = os.path.join(
        os.environ["WORKDIR"],
        "src",
        "analysis",
        "evaluation",
    )
    sys.path.append(analysis_eval_path)
    from category import CategoricalModel  # noqa

    category_model = CategoricalModel()
    dataset = "DEAM"
    testdata_for_DEAM = get_emotion(dataset)
    testdata_for_DEAM = testdata_for_DEAM.mean(axis=1)
    category_model.category_to_onehot(
        va_data=testdata_for_DEAM,
        is_save=True,
        optinal_dir=dataset,
    )

    dataset_path = os.path.join(
        os.environ["WORKDIR"], "src", "data", "DEAM", "dataset"
    )
    metadata = get_metadata(dataset_path=dataset_path, seed=seed)
    music_id = get_music_id(dataset_path=dataset_path, seed=seed)

    print("[VA]", testdata_for_DEAM[:5])
    print("[music metadata]", metadata[:5])
    print("[music_id]", music_id[:5])


if __name__ == "__main__":
    main()
