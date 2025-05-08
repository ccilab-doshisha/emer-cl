import typing as t

import numpy as np
from nptyping import Float, NDArray

"""
    cosine similarity
    arr1 : tiled vector
    arr2 : target array

    ex)
    arr1 = [[0.1, 0.3], [ 0.1, 0.3], [ 0.1,  0.3]]
    arr2 = [[0.4, 0.5], [-0.1, 0.3], [-0.5, -0.4]]
    >>
    cos_sim = [0.93834312, 0.8, -0.83957016]
"""


def cos_sim(arr1: NDArray, arr2: NDArray) -> NDArray:
    eps = 1e-16
    term1 = np.sum(arr1 * arr2, axis=1)
    term2 = np.linalg.norm(arr1, axis=1) * np.linalg.norm(arr2, axis=1)
    return term1 / (term2 + eps)


"""
    mean_average_precision
    classes : sorted classes array
    corrects : lists of correct indices

    ex)
    l1 = [1, 2, 1, 1, 2, 3, 3, 1]
    l2 = [1, 2, 3, 3, 1, 2, 2, 2]
    l3 = [1, 1, 1, 2, 3, 1, 1, 3]
    classes = [l1, l2, l3]
    corrects = [1, 2, 3]
    >>
    mAP = 0.46488095238095245
    AP = [0.7291666666666666, 0.44047619047619047, 0.225]
"""


def mAP(
    classes: t.List[t.List[int]], corrects: t.List[int]
) -> t.Union[Float, t.List[Float]]:

    """correct class index to one"""

    def trans_class_to_one(class_list: t.List[int], correct: int) -> NDArray:
        class_list_arr: NDArray = np.asarray(class_list)
        return np.where(class_list_arr == correct, 1, 0)

    """precision at k"""

    def precision_at_k(r: NDArray, k: int) -> Float:
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError("Relevance score length < k")
        # NOTE: numpy takes the average of dtype as 1 for true and 0 for false
        return np.mean(r)

    """average_precision"""

    def average_precision(r: NDArray) -> Float:
        r_bool = r != 0
        res = [
            precision_at_k(r_bool, k + 1)
            for k in range(r_bool.size)
            if r_bool[k]
        ]
        if not res:
            return 0.0
        return np.mean(res)

    # rs : rank arrays where only the correct class is set to 1
    rs = [
        trans_class_to_one(class_list, correct)
        for class_list, correct in zip(classes, corrects)
    ]

    APs = [average_precision(r) for r in rs]
    # mAP, APs
    return [np.mean(APs), APs]


"""
    entropies at k
    Sum(Plog2P), Pi = (The [num of samples] of the i-th category) / ([num of samples])

    ex)
    l1 = [1, 2, 1, 1, 2, 3, 3, 1]
    l2 = [1, 2, 3, 3, 1, 2, 2, 2]
    l3 = [1, 1, 1, 2, 3, 1, 1, 3]
    classes = [l1, l2, l3]

    k=5: [1.4591479170272448, 1.584962500721156, 1.2516291673878228]
"""


def entropies(
    classes: t.List[t.List[int]], k: t.Optional[int] = None
) -> t.List[Float]:
    def entropy(category: t.List[int], k: int) -> Float:
        category = np.asarray(category)[: k + 1]
        C = np.unique(category)
        C_num = np.asarray([np.count_nonzero(category == c) for c in C])
        probabilities = C_num / len(category)
        entropy = -np.sum([p * np.log2(p) for p in probabilities])

        # When -0.0
        if np.signbit(entropy):
            return -entropy

        return entropy

    if k is None:
        k = len(classes[0])

    entropies = [entropy(category, k) for category in classes]
    return entropies
