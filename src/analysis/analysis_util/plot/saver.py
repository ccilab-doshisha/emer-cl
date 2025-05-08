import os
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .plotter import plotter

"""
ex)
eval_type: m2e or e2m
measure_type: cosine_similarity,
eval_data: e2m_data['sim']
"""


def pdf_saver(
    eval_type: str,
    query_type: str,  # music or emotion
    measure_type: str,
    save_dir: str,
    eval_data: t.List[float],
) -> None:

    fig: Figure = plotter(
        eval_data=eval_data,
        xlabel=(query_type + " query"),
        ylabel=measure_type.replace("_", " "),
    )
    eval_data_mean = np.mean(eval_data)
    eval_data_std = np.std(eval_data)

    # save eval result
    with open(os.path.join(save_dir, f"result_{eval_type}.txt"), mode="w") as f:
        f.write("mean: " + str(eval_data_mean) + "\n")
        f.write("std: " + str(eval_data_std) + "\n")

    pp = PdfPages(os.path.join(save_dir, f"analysis_{eval_type}.pdf"))
    pp.savefig(fig)
    pp.close()
    plt.clf()
