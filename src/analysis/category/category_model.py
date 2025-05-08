import math
import os
import typing as t

import numpy as np
import tensorflow as tf
from category import CATEGORIES
from nptyping import NDArray

category_path = os.path.join(
    os.environ["WORKDIR"], "src", "analysis", "category"
)


class CategoricalModel(object):
    """
    Ref.
    I. Dufour and G. Tzanetakis,
    Using circular models to improve music emotion recognition,
    IEEE Trans. Affect. Comput., 2018.

    Beginning at 0°, 40 pieces, 5 categories
    """

    def __init__(self) -> None:
        categories, category_angles = self._setup()
        self.categories = categories
        self.category_angles = category_angles
        self.category_angles_flat = category_angles.reshape(-1)

    def _setup(self) -> t.List[NDArray]:
        # npy-dir check
        npy_dir: str = os.path.join(category_path, "npy")
        if not os.path.isdir(npy_dir):
            os.makedirs(npy_dir)

        # categories.npy check
        categories = np.array(CATEGORIES)
        categories_path: str = os.path.join(npy_dir, "categories.npy")
        if not os.path.isfile(categories_path):
            np.save(categories_path, categories)

        # CHANGE
        # categorical terms
        categories_flat_path: str = os.path.join(npy_dir, "categories_flat.npy")
        if not os.path.isfile(categories_flat_path):
            categories_flat = categories.reshape(-1)
            np.save(categories_flat_path, categories_flat)

        # category_angles.npy check
        category_angles: NDArray
        category_angles_path: str = os.path.join(npy_dir, "category_angles.npy")
        if os.path.isfile(category_angles_path):
            category_angles = np.load(category_angles_path)
        else:
            category_angles = [
                [categories1 for categories1 in range(90, (27 - 1), -9)],
                [categories2 for categories2 in range(18, (-45 - 1), -9)],
                [categories3 for categories3 in range(-54, (-117 - 1), -9)],
                [categories4 for categories4 in range(-126, (-171 - 1), -9)],
                [categories5 for categories5 in range(162, (99 - 1), -9)],
            ]
            # Border
            category_angles[3].append(180)
            category_angles[3].append(171)
            category_angles = np.array(category_angles)
            np.save(category_angles_path, category_angles)

        return [categories, category_angles]

    def _va_to_category(
        self, va_data: NDArray, is_save: bool = False, optinal_dir: str = ""
    ) -> t.List[NDArray]:

        category_index: t.List[int] = []
        # 0 to 4 classes
        category_classes: t.List[int] = []

        for valence, arousal in va_data:

            angle = int(math.degrees(math.atan2(arousal, valence)))
            if not np.any(self.category_angles_flat == angle):
                #  offset
                offset = (angle % 9) + 9
                angle = angle - offset
                if angle <= -180:
                    angle += 180

            category_index.append(
                np.where(self.category_angles_flat == angle)[0][0]
            )

            category_classes.append(
                [
                    i
                    for i, _angle in enumerate(self.category_angles)
                    if np.any(_angle == angle)
                ][0]
            )

            if is_save:
                save_dir: str = os.path.join(category_path, "npy", optinal_dir)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                np.save(
                    os.path.join(save_dir, "category_index"),
                    np.array(category_index),
                )
                np.save(
                    os.path.join(save_dir, "category_classes"),
                    np.array(category_classes),
                )

        category_index = np.array(category_index)
        category_classes = np.array(category_classes)

        return [category_index, category_classes]

    def category_to_onehot(
        self,
        va_data: NDArray,
        is_keep: bool = False,
        is_save: bool = False,
        optinal_dir: str = "",
    ) -> NDArray:
        category_index, category_classes = self._va_to_category(
            va_data, is_save=is_save, optinal_dir=optinal_dir
        )
        category_onehot = tf.keras.utils.to_categorical(category_index)
        if is_keep:
            self.category_index = category_index
            self.category_onehot = category_onehot
            self.category_classes = category_classes

        if is_save:
            save_dir: str = os.path.join(category_path, "npy", optinal_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            np.save(os.path.join(save_dir, "category_onehot"), category_onehot)

            np.save(
                os.path.join(save_dir, "category_classes"), category_classes
            )

        return category_onehot, category_classes

    def category_onehot_loader(
        self,
        optinal_dir: str,
    ) -> t.Optional[t.List[NDArray]]:

        onehot_path: str = os.path.join(
            category_path, "npy", "category_onehot.npy"
        )

        classes_path: str = os.path.join(
            category_path, "npy", "category_classes.npy"
        )

        if not (os.path.isfile(onehot_path) and os.path.isfile(classes_path)):
            return None

        category_onehot = np.load(onehot_path)
        category_classes = np.load(classes_path)

        return [category_onehot, category_classes]

    def category_index_loader(
        self,
        optinal_dir: str,
    ) -> t.Optional[NDArray]:
        index_path: str = os.path.join(
            category_path, "npy", "category_index.npy"
        )
        if not os.path.isfile(index_path):
            return None

        category_index = np.load(index_path)
        return category_index


"""For check"""


def main() -> None:
    import sys  # noqa

    sys.path.append(os.path.join(os.environ["WORKDIR"], "src"))
    from dataset import load_split_seed, load_va  # noqa

    load_split_seed(os.path.join(category_path, "sample_seed"))

    category_model = CategoricalModel()
    # DEAM
    dataset = "DEAM"
    _, testdata_for_DEAM = load_va(dataset)
    testdata_for_DEAM = testdata_for_DEAM.mean(axis=1)
    category_onehot, category_classes = category_model.category_to_onehot(
        va_data=testdata_for_DEAM,
        is_keep=True,
        is_save=True,
        optinal_dir=dataset,
    )
    print("[VA] %s example :" % dataset, testdata_for_DEAM[:5])
    print(
        "[category_index] %s example :" % dataset,
        category_model.category_index[:5],
    )
    print("[category_onehot] %s example :" % dataset)
    for i, onehot in enumerate(category_onehot[:5]):
        print("%d >>" % i, onehot)
    print("[category_classes] %s example :" % dataset, category_classes[:5])


if __name__ == "__main__":
    main()
