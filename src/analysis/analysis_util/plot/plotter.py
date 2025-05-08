import typing as t

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plotter(
    eval_data: t.List[float],
    xlabel: str,
    ylabel: str,
    figsize: t.Tuple[int] = (30, 30),
    font_size: int = 40,
    position: int = 211,  # reported results using 221
) -> Figure:

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(position)

    plt.xticks(fontsize=(font_size - 5))
    plt.yticks(fontsize=(font_size - 5))
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    ax.grid(axis="y", c="gainsboro", zorder=9)
    ax.bar(range(len(eval_data)), eval_data)

    return fig
