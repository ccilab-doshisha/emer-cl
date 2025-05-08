import argparse
import itertools
import os
import typing as t

import numpy as np
from analysis_util import load_json_as_dict
from analysis_util.plot import (
    average_precision_plot,
    cos_sim_plot,
    entropies_plot,
)

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


# data -> {key1: list, key2: list, ...}
# measure_key -> "sim", "APs", "entropy"
def _merge_result_over_keys(
    data: t.Dict[str, t.Any], measure_key: str
) -> t.Dict[str, t.List[float]]:
    res_list: t.List[float] = []

    for key in data:
        res_list.append(data[key][measure_key])

    if type(res_list[0]) is list:
        # flatten
        res_list = list(itertools.chain.from_iterable(res_list))  # noqa

    res_dict: t.Dict[str, t.Any] = {measure_key: res_list}

    return res_dict


def main() -> None:

    result_dict: t.Optional[t.Dict[str, t.Any]] = load_json_as_dict(
        dataset_name=params.dataset,
        filename="analysed_result.json",
    )
    if not result_dict:
        print(
            "You will need to run ``detailed_analysis.py''",
            "to create ``result_dict.json''. ",
        )
        return

    fig_dir: str = os.path.join(
        os.environ["WORKDIR"],
        "src",
        "analysis",
        "figure",
        params.dataset,
    )
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    # cos_sim
    cos_sim_m2e = _merge_result_over_keys(
        data=result_dict["m2e"], measure_key="sim"
    )

    cos_sim_e2m = _merge_result_over_keys(
        data=result_dict["e2m"], measure_key="sim"
    )

    cos_sim_plot(
        data={"m2e": cos_sim_m2e, "e2m": cos_sim_e2m}, base_dir=fig_dir
    )

    if params.dataset == "DEAM":
        APs = _merge_result_over_keys(
            data=result_dict["e2m"], measure_key="APs"
        )
        # Add mAP
        mAP = _merge_result_over_keys(
            data=result_dict["e2m"], measure_key="mAP"
        )
        APs.update({"mAP": np.mean(mAP["mAP"])})

        entropies = _merge_result_over_keys(
            data=result_dict["e2m"], measure_key="entropy"
        )

        average_precision_plot(data=APs, base_dir=fig_dir)
        entropies_plot(data=entropies, base_dir=fig_dir)


if __name__ == "__main__":
    main()
