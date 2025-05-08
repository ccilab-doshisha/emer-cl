import os
import typing as t

from .saver import pdf_saver


def cos_sim_plot(data: t.Dict[str, t.List[t.Any]], base_dir: str) -> None:
    measure_type = "cosine_similarity"
    sim_m2e = data["m2e"]["sim"]
    sim_e2m = data["e2m"]["sim"]
    save_dir = os.path.join(base_dir, measure_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pdf_saver(
        eval_type="m2e",
        query_type="music",
        measure_type=measure_type,
        save_dir=save_dir,
        eval_data=sim_m2e,
    )
    pdf_saver(
        eval_type="e2m",
        query_type="emotion",
        measure_type=measure_type,
        save_dir=save_dir,
        eval_data=sim_e2m,
    )


def average_precision_plot(
    data: t.Dict[str, t.List[t.Any]], base_dir: str
) -> None:
    measure_type = "average_precision"
    APs_e2m = data["APs"]
    save_dir = os.path.join(base_dir, measure_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pdf_saver(
        eval_type="e2m",
        query_type="emotion",
        measure_type=measure_type,
        save_dir=save_dir,
        eval_data=APs_e2m,
    )
    # mAP will be saved
    with open(os.path.join(save_dir, "result_e2m.txt"), mode="a") as f:
        f.write("mAP: " + str(data["mAP"]) + "\n")


def entropies_plot(data: t.Dict[str, t.List[t.Any]], base_dir: str) -> None:
    measure_type = "entropies"
    entropies_e2m = data["entropy"]
    save_dir = os.path.join(base_dir, measure_type)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pdf_saver(
        eval_type="e2m",
        query_type="emotion",
        measure_type=measure_type,
        save_dir=save_dir,
        eval_data=entropies_e2m,
    )
