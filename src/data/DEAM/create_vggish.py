import os
import sys
import typing as t

import numpy as np
import tensorflow as tf
from tqdm import tqdm

BASE_: str = os.path.join(os.environ["WORKDIR"], "src", "data")
BASE: str = os.path.join(BASE_, "DEAM")
DATASET: str = os.path.join(BASE, "dataset")
AUDIO: str = os.path.join(DATASET, "MEMD_audio")
VGGISH: str = os.path.join(BASE_, "vggish")
sys.path.append(VGGISH)

import vggish_input  # noqa
import vggish_params  # noqa
import vggish_slim  # noqa


def main() -> None:

    if os.path.isfile(os.path.join(BASE, "vggish_feat.npy")):
        return

    examples_batches: t.List[t.Any] = []
    file_nums: t.List[int] = sorted(
        [int(f.name.split(".")[0]) for f in os.scandir(AUDIO) if f.is_file()]
    )

    # checkpoint
    if os.path.isfile(os.path.join(BASE, "log_mel_examples.npy")):
        examples_batches = np.load(
            os.path.join(BASE, "log_mel_examples.npy")
        ).tolist()
    else:
        for num in tqdm(file_nums):
            wav_file = os.path.join(AUDIO, "wav", "audio_%d.wav" % (num))
            examples_batches.append(vggish_input.wavfile_to_examples(wav_file))
        np.save(
            os.path.join(BASE, "log_mel_examples"), np.array(examples_batches)
        )

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
            vggish_feat.append(embedding_batch.tolist())
            print("\033[1;32m[LOG]\033[0m #{} done. ".format(i + 1))

        vggish_feat = np.array(vggish_feat)

        # save
        np.save(os.path.join(BASE, "vggish_feat"), vggish_feat)
        print("\033[1;32m[LOG]\033[0m vggish_sequence is saved.")


if __name__ == "__main__":
    main()
