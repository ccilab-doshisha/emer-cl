import argparse
import typing as t

import numpy as np
from analysis_util import (
    cos_sim,
    entropies,
    get_emotion,
    get_emotion_classes,
    get_vggish,
    load_json_as_dict,
    mAP,
    save_dict_as_json,
    set_split_seed,
)
from category import CategoricalModel
from nptyping import NDArray

# ============================== param ==============================
parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="dataset name (DEAM or PMEmo)",
)

params = parser.parse_args()
#  ============================== param ==============================


# cos_sim helper
# model_type <- ['m2e', 'e2m']
def _cos_sim(
    test_sample: NDArray,
    sorted_indices: t.List[t.List[int]],
    correct_indices: t.List[int],
    topn: int = 5,
) -> t.List[t.Any]:

    # empty
    if not sorted_indices:
        return []

    topn_idx = int(len(test_sample) * topn / 100)
    correct_samples = [
        test_sample[indices[correct_indices[i]]].tolist()
        for i, indices in enumerate(sorted_indices)
    ]

    tiled_correct_samples = [
        np.tile(sample, reps=(len(test_sample), 1))
        for sample in correct_samples
    ]

    sim_result = [
        cos_sim(tiled_correct_samples[i], test_sample[indices])[
            :topn_idx
        ].tolist()
        for i, indices in enumerate(sorted_indices)
    ]

    return np.mean(sim_result, axis=1).tolist()


# Calculate mAP, AP, entropy, numOfchengedCategory
def _category_analysis_for_e2m(
    sorted_indices_e2m: t.List[NDArray],
    correct_indices_e2m: t.List[int],
    num_of_samples: int,
    topn: int = 5,
) -> t.Dict[str, t.Any]:

    topn_idx = int(num_of_samples * topn / 100)
    # class
    emotion_classes = get_emotion_classes(params.dataset)
    all_classes = [
        emotion_classes[indices].tolist() for indices in sorted_indices_e2m
    ]
    correct_classes = [
        emotion_classes[indices[correct_indices_e2m[i]]]
        for i, indices in enumerate(sorted_indices_e2m)
    ]

    mAP_result, APs_result = mAP(all_classes, correct_classes)
    entropy_result = entropies(all_classes, topn_idx)

    return {
        "mAP": mAP_result,
        "APs": APs_result,
        "entropy": entropy_result,
    }


def main() -> None:
    # load result data
    classified_result: t.Optional[t.Dict[str, t.Any]] = load_json_as_dict(
        dataset_name=params.dataset,
        filename="classified_result.json",
    )
    if not classified_result:
        print(
            "You will need to run ``create_eval.py''",
            "to create ``classified_result.json''. ",
        )
        return

    set_split_seed(classified_result["split_seed"])

    m2e_results = classified_result["m2e"]
    e2m_results = classified_result["e2m"]

    # =======================================
    va_data = get_emotion(params.dataset)
    vggish_data = get_vggish(params.dataset)
    m2e_eval_result: t.Dict[str, t.Any] = {}
    e2m_eval_result: t.Dict[str, t.Any] = {}
    # =======================================

    keys = ["good", "middle", "bad"]

    for key in keys:

        # <<<<<<<<<<<<<<<<<<<<< Preparation >>>>>>>>>>>>>>>>>>>

        """NOTE
        sorted_indices: Sorted indexes
        correct_va: Correct emotions

        ``sorted_indices'' is the actual index,
        and ``correct_indices'' is the correct sample position.
        """

        # For M2E
        sorted_indices_m2e = [
            result["indexes"] for result in m2e_results[f"{key}_results"]
        ]

        correct_indices_m2e = [
            result["rank"] - 1 for result in m2e_results[f"{key}_results"]
        ]
        # For E2M
        sorted_indices_e2m = [
            result["indexes"] for result in e2m_results[f"{key}_results"]
        ]
        correct_indices_e2m = [
            result["rank"] - 1 for result in e2m_results[f"{key}_results"]
        ]
        # <<<<<<<<<<<<<<<<<<<<< Preparation >>>>>>>>>>>>>>>>>>>

        # <<<<<<<<<<<<<<<<<<<<< Running the analysis >>>>>>>>>>>>>>>>>>>

        # M2E
        sim_m2e = _cos_sim(
            test_sample=vggish_data,
            sorted_indices=sorted_indices_m2e,
            correct_indices=correct_indices_m2e,
        )

        # E2M
        sim_e2m = _cos_sim(
            test_sample=va_data,
            sorted_indices=sorted_indices_e2m,
            correct_indices=correct_indices_e2m,
        )

        m2e_eval_result[key] = {"sim": sim_m2e}

        e2m_eval_result[key] = {"sim": sim_e2m}

        """NOTE
        Only DEAM supports analysis with a variety of discrete data
        """
        if params.dataset == "DEAM":
            # saved categorical model
            categorical_model = CategoricalModel()
            categorical_model.category_to_onehot(
                va_data=va_data,
                is_keep=True,
                is_save=True,
                optinal_dir=params.dataset,
            )
            categorical_results: t.Dict[
                str, t.Any
            ] = _category_analysis_for_e2m(
                sorted_indices_e2m=sorted_indices_e2m,
                correct_indices_e2m=correct_indices_e2m,
                num_of_samples=len(va_data),
            )
            e2m_eval_result[key].update(
                {
                    "mAP": categorical_results["mAP"],
                    "APs": categorical_results["APs"],
                    "entropy": categorical_results["entropy"],
                }
            )
        # <<<<<<<<<<<<<<<<<<<<< Running the analysis >>>>>>>>>>>>>>>>>>>

    result_dict = {"m2e": m2e_eval_result, "e2m": e2m_eval_result}

    save_dict_as_json(
        save_dict=result_dict,
        dataset_name=params.dataset,
        filename="analysed_result.json",
    )


if __name__ == "__main__":
    main()
