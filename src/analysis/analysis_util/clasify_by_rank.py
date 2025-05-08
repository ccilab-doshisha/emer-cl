import typing as t

from nptyping import NDArray

from .utils import get_music_id


def _clasify(
    eval_result: t.Dict[str, t.Any], music_ids: NDArray
) -> t.Dict[str, t.Any]:
    """
    Good : Top10 -> rank <= len(testdata) * 0.10
    Bad : Under 50% -> rank >= len(testdata) * 0.50
    Middle : 10% ~ 50% else
    """
    num_of_sample = len(music_ids)
    sorted_indexes = eval_result["indexes"]
    ranks = eval_result["ranks"]

    # === Sort the queries in order of decreasing rank value ===
    queries_rank = eval_result["queries_rank"]
    sorted_indexes = sorted_indexes[queries_rank]
    ranks = ranks[queries_rank]
    # ==========================================================

    border_top = num_of_sample * 0.10
    border_bottom = num_of_sample * 0.50

    good_results: t.List[t.Dict[str, t.Any]] = []
    middle_results: t.List[t.Dict[str, t.Any]] = []
    bad_results: t.List[t.Dict[str, t.Any]] = []

    for rank, indexes, music_id in zip(ranks, sorted_indexes, music_ids):
        filenumbers = [music_ids[num] for num in indexes]
        if rank <= border_top:
            good_results.append(
                {
                    "rank": int(rank),
                    "filenumber": music_id,
                    "indexes": indexes.tolist(),
                    "filenumbers": filenumbers,
                }
            )

        elif rank >= border_bottom:
            bad_results.append(
                {
                    "rank": int(rank),
                    "filenumber": music_id,
                    "indexes": indexes.tolist(),
                    "filenumbers": filenumbers,
                }
            )

        else:
            middle_results.append(
                {
                    "rank": int(rank),
                    "filenumber": music_id,
                    "indexes": indexes.tolist(),
                    "filenumbers": filenumbers,
                }
            )

    good_num = len(good_results)
    bad_num = len(bad_results)
    middle_num = len(middle_results)

    result = {
        "good_results": good_results,
        "middle_results": middle_results,
        "bad_results": bad_results,
        "num": {
            "good_num": good_num,
            "middle_num": middle_num,
            "bad_num": bad_num,
        },
        "border": {"top": border_top, "bottom": border_bottom},
        "AR": eval_result["AR"],
        "MRR": eval_result["MRR"],
    }
    return result


def clasify_by_rank(
    eval_result: t.Dict[str, t.Any],
    dataset_path: str,
    split_seed: int,
) -> t.Dict[str, t.Any]:

    test_music_ids = get_music_id(dataset_path=dataset_path, seed=split_seed)

    classified_m2e = _clasify(
        eval_result=eval_result["m2e"], music_ids=test_music_ids
    )
    classified_e2m = _clasify(
        eval_result=eval_result["e2m"], music_ids=test_music_ids
    )
    classified_result = {"m2e": classified_m2e, "e2m": classified_e2m}

    return classified_result
