import argparse
import os
import typing as t

import numpy as np
import tensorflow as tf
from dataset import load_split_seed, load_va, load_vggish
from evaluation import evaluation
from model import EMER_CL
from nptyping import NDArray
from util import LinearCCA, load_params

# ============================== param ==============================
parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="dataset name (DEAM or PMEmo)",
)

parser.add_argument(
    "--datetime",
    "--model_dir",
    default=None,
    type=str,
    required=True,
    help="datetime string (used as dirname, ex. 2021_0901_12_30_55)",
)

parser.add_argument(
    "--corr_p",
    default=None,
    type=float,
    help="threshold for correlation similarity in evaluation (need float)",
)

params = parser.parse_args()
target_dir = os.path.join(
    os.environ["WORKDIR"], "src", "model", params.dataset, params.datetime
)
train_params = load_params(target_dir)
load_split_seed(target_dir)
#  ============================== param ==============================


def restore(
    sess: tf.compat.v1.Session,
    model: EMER_CL,
    l_cca: LinearCCA,
    train_samples: t.List[NDArray],
) -> t.List[t.Union[NDArray, None]]:

    [vggish_train, va_train] = train_samples
    output_tensors = model.get_output_tensors()
    # restore
    saver = tf.compat.v1.train.Saver()
    ckpt_path = os.path.join(target_dir, "model.ckpt")
    saver.restore(sess, ckpt_path)

    # ======= calcurate for approximation line using training sample ======
    if train_params._lambda != 0:
        f1, f2 = sess.run(
            [
                output_tensors["music_enc"]["emb"],
                output_tensors["emotion_enc"]["emb"],
            ],
            feed_dict={
                model.input_music: vggish_train,
                model.input_emotion: va_train,
            },
        )
        l_cca.fit(f1, f2)

        # CCA output
        f1_hat, f2_hat = l_cca.test(f1_hat=f1, f2_hat=f2)
        # Calculating correlation
        num = len(f1_hat.T)
        corrs_all = np.corrcoef(f1_hat.T[:num], f2_hat.T[:num])[:num, num:]
        train_corrs = np.diag(corrs_all)
        # Calculating approximation
        line_params_m2e = np.array(
            [
                np.polyfit(dim1, dim2, 1).tolist()
                for dim1, dim2 in zip(f1_hat.T, f2_hat.T)
            ]
        )
        line_params_e2m = np.array(
            [
                np.polyfit(dim2, dim1, 1).tolist()
                for dim1, dim2 in zip(f1_hat.T, f2_hat.T)
            ]
        )
        """
        train_corrs: Correlation in training samples in each dimension.
        line_params_m2e: lists in weight and bias for m2e
        line_params_e2m: lists in weight and bias for e2m
        """
        return [train_corrs, line_params_m2e, line_params_e2m]

    # Return a dummy values when only KL divergence is used.
    return [None, None, None]


def main() -> None:

    """setting"""
    tf.compat.v1.reset_default_graph()

    # load dataset
    vggish_train, vggish_test = load_vggish(params.dataset)
    va_train, va_test = load_va(params.dataset)

    # model
    model = EMER_CL(
        train_params.dataset,
        train_params.embedding_dim,
    )
    l_cca = LinearCCA(train_params.embedding_dim)

    tf_config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )
    sess = tf.compat.v1.Session(config=tf_config)

    """restore
    Return approximation-line parameters for the training samples
    will be used in the validation
    """
    train_corrs, line_params_m2e, line_params_e2m = restore(
        sess=sess,
        model=model,
        l_cca=l_cca,
        train_samples=[vggish_train, va_train],
    )

    """evaluation"""
    output_tensors = model.get_output_tensors()
    output_music: t.Dict[str, tf.Tensor]
    output_emotion: t.Dict[str, tf.Tensor]
    output_music, output_emotion = sess.run(
        [output_tensors["music_enc"], output_tensors["emotion_enc"]],
        feed_dict={
            model.input_music: vggish_test,
            model.input_emotion: va_test,
        },
    )

    if train_params._lambda == 0:
        f1_hat, f2_hat = 0.0, 0.0
    else:
        f1_hat, f2_hat = l_cca.test(
            f1_hat=output_music["emb"], f2_hat=output_emotion["emb"]
        )

    corr_p = params.corr_p if params.corr_p else train_params.corr_p

    evaluation(
        f1_hat=f1_hat,
        f2_hat=f2_hat,
        train_corrs=train_corrs,
        approximate_lines_params=[line_params_m2e, line_params_e2m],
        music_dist=[output_music["mu"], output_music["log_sigma"]],
        emotion_dist=[output_emotion["mu"], output_emotion["log_sigma"]],
        _lambda=train_params._lambda,
        corr_p=corr_p,
    )

    sess.close()


if __name__ == "__main__":
    main()
