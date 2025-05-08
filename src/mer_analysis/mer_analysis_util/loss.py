import typing as t
from operator import itemgetter

import tensorflow as tf


class CCALoss(object):
    def __init__(self, dim: int):
        shape = [dim, dim]
        self.W1 = tf.Variable(
            tf.random.truncated_normal(shape, stddev=0.1),
            name="embedding_weight_v1",
        )
        self.W2 = tf.Variable(
            tf.random.truncated_normal(shape, stddev=0.1),
            name="embedding_weight_v2",
        )

    def loss(self, f1: tf.Tensor, f2: tf.Tensor) -> tf.Tensor:

        r1 = tf.constant(1e-4)
        r2 = tf.constant(1e-4)

        m = tf.shape(f1)[0]
        o1 = o2 = tf.shape(f1)[1]

        f1 = tf.transpose(f1)
        f2 = tf.transpose(f2)

        m1 = tf.reduce_mean(f1, axis=0, keep_dims=True)
        f1_bar = tf.subtract(f1, m1)
        m2 = tf.reduce_mean(f2, axis=0, keep_dims=True)
        f2_bar = tf.subtract(f2, m2)

        one = tf.constant([1.0])
        m_float = tf.cast(m, "float")

        partition = tf.divide(one, (m_float - 1))
        sigma_hat11 = partition * tf.matmul(f1_bar, tf.transpose(f1_bar))
        sigma_hat11 += r1 * tf.eye(o1)
        sigma_hat22 = partition * tf.matmul(f2_bar, tf.transpose(f2_bar))
        sigma_hat22 += r2 * tf.eye(o2)
        sigma_hat12 = partition * tf.matmul(f1_bar, tf.transpose(f2_bar))

        corr = tf.matmul(tf.matmul(tf.transpose(self.W1), sigma_hat12), self.W2)
        corr = tf.matmul(corr, tf.transpose(corr))
        corr = tf.linalg.trace(tf.sqrt(corr))

        subject_to_1 = tf.eye(o1) - tf.matmul(
            tf.matmul(tf.transpose(self.W1), sigma_hat11), self.W1
        )
        subject_to_2 = tf.eye(o2) - tf.matmul(
            tf.matmul(tf.transpose(self.W2), sigma_hat22), self.W2
        )
        subject = tf.abs(subject_to_1 + subject_to_2) * 0.5

        loss = -1.0 * corr + tf.reduce_sum(subject)
        return loss


class RankLoss(object):
    def __init__(self, margin: float):
        self.margin = margin

    # KL_divergence
    def _kld(
        self,
        mu0: tf.Tensor,
        sigma0: tf.Tensor,
        mu1: tf.Tensor,
        sigma1: tf.Tensor,
    ) -> tf.Tensor:
        # ------------ preparation ------------
        eps = 1e-6
        sigma0 = sigma0 + eps
        sigma1 = sigma1 + eps
        dim = tf.shape(mu0)[1]
        tile_shape = [tf.shape(mu0)[0], tf.shape(mu0)[0]]
        sigma1_inv = 1.0 / sigma1
        # -----------------------------------------

        # term1
        t1 = sigma1_inv @ tf.transpose(sigma0)

        # term2
        t2_1 = tf.reduce_sum(mu1 * mu1 * sigma1_inv, axis=1)
        t2_1 = tf.transpose(tf.broadcast_to(t2_1, tile_shape))
        t2_2 = (mu1 * sigma1_inv) @ tf.transpose(mu0)
        t2_3 = sigma1_inv @ tf.transpose(mu0 * mu0)
        t2 = t2_1 - (2.0 * t2_2) + t2_3

        # term3 -> only dim
        t3 = tf.cast(dim, tf.float32)

        # term4
        t4_1 = tf.reduce_sum(tf.math.log(sigma1), axis=1)
        t4_2 = tf.reduce_sum(tf.math.log(sigma0), axis=1)
        t4_1 = tf.transpose(tf.broadcast_to(t4_1, tile_shape))
        t4_2 = tf.broadcast_to(t4_2, tile_shape)
        t4 = t4_1 - t4_2

        kld = (t1 + t2 - t3 + t4) * 0.5
        return kld

    def loss(
        self, music_dist: t.List[tf.Tensor], emotion_dist: t.List[tf.Tensor]
    ) -> tf.Tensor:

        # ------------ preparation ------------
        mu0, log_sigma0 = music_dist
        mu1, log_sigma1 = emotion_dist
        norm_rate = 1e-3
        # -----------------------------------------

        sigma0 = tf.math.exp(log_sigma0)
        sigma1 = tf.math.exp(log_sigma1)

        scores_m = self._kld(mu0, sigma0, mu1, sigma1)
        scores_e = self._kld(mu1, sigma1, mu0, sigma0)
        positive_m = tf.expand_dims(tf.linalg.tensor_diag_part(scores_m), 1)
        positive_e = tf.expand_dims(tf.linalg.tensor_diag_part(scores_e), 1)

        zero = tf.add(tf.cast(tf.zeros(tf.shape(mu0)[0]), tf.float32), 0.0)
        cost_m = tf.maximum(zero, positive_m - scores_m + self.margin)
        cost_e = tf.maximum(zero, positive_e - scores_e + self.margin)
        cost_m = tf.linalg.set_diag(cost_m, zero)
        cost_e = tf.linalg.set_diag(cost_e, zero)

        # square & sqrt
        norm_m = tf.math.scalar_mul(
            norm_rate, tf.math.sqrt(tf.math.square(positive_m))
        )
        norm_e = tf.math.scalar_mul(
            norm_rate, tf.math.sqrt(tf.math.square(positive_e))
        )

        return tf.reduce_sum(cost_m + norm_m + cost_e + norm_e)


class CompositeLoss(object):

    cca_loss: CCALoss
    rank_loss: RankLoss

    def __init__(self, _lambda: float, margin: float, dim: int):
        self._lambda = _lambda
        self.cca_loss = CCALoss(dim)
        self.rank_loss = RankLoss(margin)

    def call(
        self,
        music_enc_out: t.Dict[str, tf.Tensor],
        emotion_enc_out: t.Dict[str, tf.Tensor],
    ) -> tf.Tensor:

        music_dist = list(itemgetter("mu", "log_sigma")(music_enc_out))
        emotion_dist = list(itemgetter("mu", "log_sigma")(emotion_enc_out))

        if self._lambda == 0.0:
            loss = self.rank_loss.loss(music_dist, emotion_dist)
        elif self._lambda == 1.0:
            loss = self.cca_loss.loss(
                music_enc_out["emb"], emotion_enc_out["emb"]
            )
        else:
            loss = self._lambda * self.cca_loss.loss(
                music_enc_out["emb"], emotion_enc_out["emb"]
            )
            loss += (1 - self._lambda) * self.rank_loss.loss(
                music_dist, emotion_dist
            )
        return loss
