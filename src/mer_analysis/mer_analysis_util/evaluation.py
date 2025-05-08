import typing as t

import numpy as np
from nptyping import NDArray


# kl divergence
def kld(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, dim: int
) -> NDArray:
    eps = 1e-6
    sigma1 = sigma1 + eps
    sigma2 = sigma2 + eps
    sigma1_inv = 1.0 / sigma2

    t1 = sigma1_inv @ sigma1.T

    t2_1 = np.sum(mu2 * mu2 * sigma1_inv, axis=1)
    t2_1 = np.tile(t2_1, (t2_1.shape[0], 1)).T
    t2_2 = (mu2 * sigma1_inv) @ mu1.T
    t2_3 = sigma1_inv @ (mu1 * mu1).T

    t2 = t2_1 - (2.0 * t2_2) + t2_3

    t3 = dim

    t4_1 = np.sum(np.log(sigma2), axis=1)
    t4_2 = np.sum(np.log(sigma1), axis=1)
    t4_1 = np.tile(t4_1, (t4_1.shape[0], 1)).T
    t4_2 = np.tile(t4_2, (t4_2.shape[0], 1))

    t4 = t4_1 - t4_2

    kld = (t1 + t2 - t3 + t4) * 0.5
    return kld


"""
Compute AE for dimensions larger than corr_th.

corr_th : -1 ~ 1
pred_from_line : y from "ax + b"
target  f1_hat or f2_hat (correct target)
train_corrs: Correlation in training samples in each dimension.
"""


def diff_upper(
    corr_th: float,
    pred_from_line: NDArray,
    target: NDArray,
    train_corrs: NDArray,
) -> NDArray:
    corr_idx = np.where(train_corrs >= corr_th)[0]
    target = target[:, corr_idx]
    # |y - y_hat|
    result = np.array(
        [
            np.abs(target - pred_from_line[i, corr_idx])
            for i in range(len(pred_from_line))
        ]
    )
    # result has not yet been weighted by correlation.
    return (train_corrs[corr_idx] * result).sum(axis=2)


def approximate_lines(
    approximate_line_params: NDArray,
) -> t.Callable[[NDArray], NDArray]:
    if type(approximate_line_params) is not np.ndarray:
        approximate_line_params = np.array(approximate_line_params)
    W, b = approximate_line_params.T  # noqa

    def _call(input_emb: NDArray) -> NDArray:
        if type(input_emb) is list:
            input_emb = np.array(input_emb)

        assert input_emb.ndim <= 2, "Required arrays less two dimensions"
        if input_emb.ndim == 2:
            return W[np.newaxis, :] * input_emb + b
        return W * input_emb + b

    return _call


# mean reciprocal rank
def MRR(ranks: NDArray) -> float:
    mrr: float = np.sum(1.0 / ranks, dtype=float) / float(len(ranks))
    return mrr


# average rank
def AR(ranks: NDArray) -> float:
    average_rank: float = np.sum(ranks, dtype=float) / float(len(ranks))
    return average_rank


# wrapper function to calurate MRR and AR
def cal_MRR_AR_from(ranks: NDArray) -> t.List[float]:
    mrr: float = MRR(ranks)
    arr: float = AR(ranks)
    return [mrr, arr]


def evaluation(
    f1_hat: NDArray,
    f2_hat: NDArray,
    train_corrs: NDArray,
    approximate_lines_params: t.List[NDArray],
    music_dist: t.List[NDArray],
    emotion_dist: t.List[NDArray],
    _lambda: float,
    get_result: bool = False,
) -> t.Optional[t.Dict[str, t.Any]]:
    [mu1, log_sigma1] = music_dist
    [mu2, log_sigma2] = emotion_dist
    # log_sigma to sigma
    sigma1 = np.exp(log_sigma1)
    sigma2 = np.exp(log_sigma2)

    # unpack
    al_param_m2e, al_param_e2m = approximate_lines_params
    # prediction from approximate lines
    pred_from_line_m2e = approximate_lines(al_param_m2e)(f1_hat)
    pred_from_line_e2m = approximate_lines(al_param_e2m)(f2_hat)
    # Define similarity as a weighted value of the correlation
    ae_weighed_corr_m2e = diff_upper(
        0.4, pred_from_line_m2e, f2_hat, train_corrs
    )
    ae_weighed_corr_e2m = diff_upper(
        0.4, pred_from_line_e2m, f1_hat, train_corrs
    )

    # ==========================================
    # kl
    kl_m2e = kld(mu1, sigma1, mu2, sigma2, mu1.shape[1])
    kl_e2m = kld(mu2, sigma2, mu1, sigma1, mu2.shape[1])

    # sim
    m2e = (_lambda * ae_weighed_corr_m2e) + ((1 - _lambda) * kl_m2e)
    e2m = (_lambda * ae_weighed_corr_e2m) + ((1 - _lambda) * kl_e2m)

    """NOTE
    indexes: indexes when ordered by increasing similarity
    ranks: a calculation of the position of the correct sample based on the indexes
    sorted_queries_by_rank: queries are ordered by decreasing ``rank values''
    """
    sorted_indexes_m2e = np.argsort(m2e, axis=1)
    ranks_m2e = np.zeros(len(sorted_indexes_m2e))

    sorted_indexes_e2m = np.argsort(e2m, axis=1)
    ranks_e2m = np.zeros(len(sorted_indexes_e2m))

    for i in range(len(sorted_indexes_m2e)):
        index_m2e = np.where(sorted_indexes_m2e[i] == i)
        ranks_m2e[i] = index_m2e[0] + 1

        index_e2m = np.where(sorted_indexes_e2m[i] == i)
        ranks_e2m[i] = index_e2m[0] + 1

    MRR_m2e, AR_m2e = cal_MRR_AR_from(ranks_m2e)
    MRR_e2m, AR_e2m = cal_MRR_AR_from(ranks_e2m)

    sorted_queries_by_rank_m2e = np.argsort(ranks_m2e)
    sorted_queries_by_rank_e2m = np.argsort(ranks_e2m)

    if get_result:
        eval_result = {
            "m2e": {
                "indexes": sorted_indexes_m2e,
                "queries_rank": sorted_queries_by_rank_m2e,
                "ranks": ranks_m2e,
                "MRR": MRR_m2e,
                "AR": AR_m2e,
            },
            "e2m": {
                "indexes": sorted_indexes_e2m,
                "queries_rank": sorted_queries_by_rank_e2m,
                "ranks": ranks_e2m,
                "MRR": MRR_e2m,
                "AR": AR_e2m,
            },
        }
        return eval_result

    print("============================== M2E ==============================")
    print(MRR_m2e)
    print(AR_m2e)
    print("=================================================================\n")

    print("============================== E2M ==============================")
    print(MRR_e2m)
    print(AR_e2m)
    print("=================================================================\n")

    return None
