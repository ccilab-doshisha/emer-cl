import argparse
import os
import sys
import typing as t

import tensorflow as tf
from analysis_util import clasify_by_rank, save_dict_as_json

sys.path.append(os.path.join(os.environ["WORKDIR"], "src"))
from dataset import get_seed, load_split_seed, load_va, load_vggish  # noqa
from restore import EMER_CL, LinearCCA, evaluation, load_params, restore  # noqa

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

params = parser.parse_args()
target_dir = os.path.join(
    os.environ["WORKDIR"], "src", "model", params.dataset, params.datetime
)
train_params = load_params(target_dir)
load_split_seed(target_dir)
split_seed = get_seed(target_dir)
#  ============================== param ==============================


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

    """restore"""
    train_corrs, line_params_m2e, line_params_e2m = restore(
        sess=sess,
        model=model,
        l_cca=l_cca,
        train_samples=[vggish_train, va_train],
    )

    """create evalation"""
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

    sess.close()

    if train_params._lambda == 0:
        f1_hat, f2_hat = 0.0, 0.0
    else:
        f1_hat, f2_hat = l_cca.test(
            f1_hat=output_music["emb"], f2_hat=output_emotion["emb"]
        )

    eval_result: t.Dict[str, t.List[t.Any]] = evaluation(
        f1_hat=f1_hat,
        f2_hat=f2_hat,
        train_corrs=train_corrs,
        approximate_lines_params=[line_params_m2e, line_params_e2m],
        music_dist=[output_music["mu"], output_music["log_sigma"]],
        emotion_dist=[output_emotion["mu"], output_emotion["log_sigma"]],
        _lambda=train_params._lambda,
        corr_p=train_params.corr_p,
        get_result=True,
    )

    dataset_path: str = os.path.join(
        os.environ["WORKDIR"], "src", "data", params.dataset, "dataset"
    )

    if params.dataset == "PMEmo":
        dataset_path = os.path.join(dataset_path, "PMEmo2019")

    """Obtain results for use in detailed analysis based on rank order"""
    classified_result: t.Dict[str, t.Any] = clasify_by_rank(
        eval_result=eval_result,
        dataset_path=dataset_path,
        split_seed=split_seed,
    )

    classified_result["split_seed"] = split_seed

    save_dict_as_json(
        save_dict=classified_result,
        dataset_name=params.dataset,
        filename="classified_result.json",
    )


if __name__ == "__main__":
    main()
