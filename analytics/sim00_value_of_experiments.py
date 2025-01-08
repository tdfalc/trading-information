import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from tfds.plotting import prettify, use_tex


def calculate_exerpiment_value(informativeness: float, type: float) -> float:
    return np.maximum(
        0,
        type * informativeness
        + 1
        + np.minimum(-informativeness, 0)
        - (1 - type) * (type <= 0.5)
        - type * (type > 0.5),
    )


if __name__ == "__main__":

    use_tex()

    savedir = Path(__file__).parent / "docs/sim00_value_of_experiments"
    os.makedirs(savedir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)

    num_types = 21
    types = np.linspace(0, 1, num_types)
    allocations = np.linspace(-1, 1, num_types)
    exerpiment_values = np.zeros((num_types, num_types))

    for i, allocation in enumerate(allocations):
        exerpiment_values[i, :] = calculate_exerpiment_value(allocation, types)

    cmap = "YlOrRd"

    im = ax.imshow(exerpiment_values[::-1], cmap=cmap, extent=[0, 4, -1, 1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cbar = fig.colorbar(im, cmap=cmap, cax=cax)
    cbar.set_label(r"Experiment Gain ($\delta$)", labelpad=10)

    ax.set_xlabel("Private Type ($v_b$)")
    ax.set_ylabel(r"Informativeness ($I$)")
    ax.set_xticks((0, 1, 2, 3, 4))
    ax.set_xticklabels(("$0.0$", "$0.25$", "$0.5$", "$0.75$", "$1.0$"))
    prettify(ax=ax)

    fig.tight_layout()
    fig.savefig(savedir / "value_of_experiments.pdf", dpi=300)
