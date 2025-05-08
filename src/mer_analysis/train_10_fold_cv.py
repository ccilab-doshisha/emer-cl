import math
import os
import typing as t
from collections import OrderedDict
from pdb import set_trace as st
from time import time

import numpy as np
from mer_analysis_util.dataset_10_fold_cv import load_va_10_folds, load_vggish_10_folds #, save_split_seed
from mer_analysis_util.evaluation import evaluation
from mer_analysis_util.loss import CompositeLoss
from mer_analysis_util.model import EMER_CL
from nptyping import NDArray
from tqdm import tqdm
from mer_analysis_util.linear_cca import LinearCCA
from mer_analysis_util.parser import parser, save_params
from mer_analysis_util.util import batch_generator, get_now


params = parser.parse_args()
# setting GPU and TensorFlow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa


def train(
    sess: tf.compat.v1.Session,
    model: EMER_CL,
    l_cca: LinearCCA,
    train_samples: t.List[NDArray],
    fold_idx: int,
) -> t.List[t.Union[NDArray, None]]:

    [vggish_train, va_train] = train_samples
    output_tensors = model.get_output_tensors()

    # loss/optimiser
    composite_loss = CompositeLoss(
        params._lambda, params.margin, params.embedding_dim
    )
    loss = composite_loss.call(
        output_tensors["music_enc"], output_tensors["emotion_enc"]
    )
    train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=params.lr
    ).minimize(loss)

    # batch_setting
    num_of_train_samples = len(vggish_train)
    mini_steps = math.ceil(num_of_train_samples / params.batch_size)
    batches = batch_generator(
        vggish_train, va_train, batch_size=params.batch_size
    )

    # model's saver
    saver = tf.compat.v1.train.Saver()

    # another setting
    output_steps = 25
    ncols = 100
    avg_loss = 0.0

    sess.run(tf.compat.v1.global_variables_initializer())
    with tqdm(range(params.epoch), ncols=ncols) as pbar:
        pbar.set_postfix(OrderedDict(loss="loss"))
        for step in pbar:  # main loop
            for mini_step in range(mini_steps):  # mini-batch loop
                (music_batch, emotion_batch) = next(batches)
                _, mini_loss = sess.run(
                    [train_op, loss],
                    feed_dict={
                        model.input_music: music_batch,
                        model.input_emotion: emotion_batch,
                    },
                )
                # last mini_step
                if mini_step == mini_steps - 1:
                    avg_loss += mini_loss

            if step > 0 and step % output_steps == 0:
                avg_loss /= output_steps
                pbar.set_postfix(OrderedDict(loss=avg_loss))
                avg_loss = 0.0

        # save_model/params
        target_dir = os.path.join(
            os.environ["WORKDIR"], "src", "model", params.dataset, str(fold_idx).zfill(2), get_now()
        )
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        saver.save(sess, os.path.join(target_dir, "model.ckpt"))
        save_params(target_dir)
        #save_split_seed(target_dir)
        
        # ======= calculate for approximation line using training sample ======
        if params._lambda != 0:
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
            dim = len(f1_hat.T)
            corrs_all: NDArray = np.corrcoef(f1_hat.T[:dim], f2_hat.T[:dim])[
                :dim, dim:
            ]
            train_corrs: NDArray = np.diag(corrs_all)
            # Calculating approximation
            line_params_m2e: NDArray = np.array(
                [
                    np.polyfit(dim1, dim2, 1).tolist()
                    for dim1, dim2 in zip(f1_hat.T, f2_hat.T)
                ]
            )
            line_params_e2m: NDArray = np.array(
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

    
    for fold_idx in range(1,11):
        
        start = time()
        print("### Training model for fold %d ..." % fold_idx)

        """setting"""
        tf.compat.v1.reset_default_graph()
    
        # load dataset
        [vggish_train, vggish_test] = load_vggish_10_folds(params.dataset,fold_idx)
        [va_train, va_test] = load_va_10_folds(params.dataset,fold_idx)        

        # model
        model = EMER_CL(
            params.dataset,
            params.embedding_dim,
        )
        l_cca = LinearCCA(params.embedding_dim)

        # another setting
        tf_config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        )   
        sess = tf.compat.v1.Session(config=tf_config)
        
        """train
        Return approximation-line parameters
        for the training samples will be used in the validation
        """
        train_corrs, line_params_m2e, line_params_e2m = train(
            sess=sess,
            model=model,
            l_cca=l_cca,
            train_samples=[vggish_train, va_train],
            fold_idx=fold_idx,
        )

        """evaluation"""
        output_tensors = model.get_output_tensors()
        output_music_test: t.Dict[str, NDArray]
        output_emotion_test: t.Dict[str, NDArray]

        output_music_test, output_emotion_test = sess.run(
            [output_tensors["music_enc"], output_tensors["emotion_enc"]],
            feed_dict={
                model.input_music: vggish_test,
                model.input_emotion: va_test,
            },
        )

        if params._lambda == 0:
            f1_hat, f2_hat = 0.0, 0.0
        else:
            f1_hat, f2_hat = l_cca.test(
                f1_hat=output_music_test["emb"], f2_hat=output_emotion_test["emb"]
            )
        # ==========================================

        evaluation(
            f1_hat=f1_hat,
            f2_hat=f2_hat,
            train_corrs=train_corrs,
            approximate_lines_params=[line_params_m2e, line_params_e2m],
            music_dist=[output_music_test["mu"], output_music_test["log_sigma"]],
            emotion_dist=[
                output_emotion_test["mu"],
                output_emotion_test["log_sigma"],
            ],
            _lambda=params._lambda,
        )

        sess.close()
        end = time()
        print("Model trained in %.2f seconds" % (end-start))


if __name__ == "__main__":
    main()
