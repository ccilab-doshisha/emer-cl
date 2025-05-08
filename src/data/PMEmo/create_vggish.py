import os
import sys
import typing as t

import numpy as np
import tensorflow as tf
from nptyping import NDArray
from tqdm import tqdm

BASE_: str = os.path.join(os.environ["WORKDIR"], "src", "data")
BASE: str = os.path.join(BASE_, "PMEmo")
DATASET: str = os.path.join(BASE, "dataset", "PMEmo2019")
AUDIO: str = os.path.join(DATASET, "chorus")
VGGISH: str = os.path.join(BASE_, "vggish")
sys.path.append(VGGISH)

import vggish_input  # noqa
import vggish_params  # noqa
import vggish_slim  # noqa


def listToNumpy(arr: t.List[t.Any]) -> NDArray:
    import warnings

    warnings.filterwarnings("error")
    try:
        return np.asarray(arr)
    # VisibleDeprecationWarning
    except Warning:
        return np.asarray(arr, dtype=object)


def main() -> None:

    if os.path.isfile(os.path.join(BASE, "vggish_feat.npy")):
        return

    examples_batches: t.List[t.Any] = []

    remain_music_indices = np.load(
        os.path.join(BASE, "remain_music_indices.npy")
    )

    for idx in tqdm(remain_music_indices):
        wav_file = os.path.join(AUDIO, "wav", "audio_%d.wav" % (idx))
        examples_batches.append(vggish_input.wavfile_to_examples(wav_file))

    print(
        "\033[1;32m[LOG]\033[0m log_mel_examples: ",
        np.array(examples_batches).shape,
    )

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Graph().as_default(), tf.Session(config=config) as sess:

        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(
            sess, os.path.join(VGGISH, "vggish_model.ckpt")
        )
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME
        )
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )

        vggish_feat: t.List[int] = []
        for i, examples_batch in enumerate(examples_batches):
            embedding_batch = sess.run(
                embedding_tensor, feed_dict={features_tensor: examples_batch}
            )
            vggish_feat.append(embedding_batch)
            print("\033[1;32m[LOG]\033[0m #{} done. ".format(i + 1))

        vggish_feat = listToNumpy(vggish_feat)

        # save
        np.save(os.path.join(BASE, "vggish_feat"), vggish_feat)
        print("\033[1;32m[LOG]\033[0m vggish_sequence is saved.")


if __name__ == "__main__":
    main()
