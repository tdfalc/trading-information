import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

logger = create_logger(__name__)


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

    logger.info("Running value of experiments analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim00_value_of_experiments"
    os.makedirs(savedir, exist_ok=True)

    # fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)
    fig, ax = plt.subplots(figsize=(4, 2.5))

    num_types = 1000
    types = np.linspace(0, 1, num_types)
    allocations = np.linspace(-1, 1, num_types)
    experiment_values = np.zeros((num_types, num_types))

    for i, allocation in enumerate(allocations):
        experiment_values[i, :] = calculate_exerpiment_value(allocation, types)

    ax.plot(types, experiment_values[500, :], color="k", lw=1, label=r"$0$")
    ax.plot(types, experiment_values[680, :], color="k", ls="dashed", lw=1, label=r"$2/5$")
    ax.plot(types, experiment_values[130, :], color="k", ls="dashdot", lw=1, label=r"$-4/5$")

    print(allocations[500], allocations[680], allocations[100])
    # for i, allocation in enumerate(allocations):
    #     experiment_values[i, :] = calculate_exerpiment_value(allocation, types)

    # ax.plot(types, experiment_values[500, :], color="k", lw=1)
    # ax.plot(types, experiment_values[750, :], color="k", ls="dashed", lw=1)

    # # Create a meshgrid
    # X, Y = np.meshgrid(types * 6, allocations * 2)

    # # Use pcolormesh instead of imshow for a gridded look
    # # cmap = "Blues"
    # # pc = ax.pcolormesh(X, Y, experiment_values, shading="auto", cmap=cmap)

    # pc = ax.pcolormesh(X, Y, experiment_values, shading="auto", cmap="Blues")

    # # Colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.3)
    # cbar = fig.colorbar(pc, cax=cax)
    # cbar.set_label(r"$\delta (I, v_b)$", labelpad=10)
    # # cbar = fig.colorbar(pc, ax=ax, orientation="vertical", pad=-0.50, aspect=40)
    # # cbar.set_label(r"$\delta (I, v_b)$", labelpad=5)

    ax.set_xlabel("$v_b$")
    # ax.set_ylabel(r"$I(v_b)$")
    ax.set_ylabel(r"$\delta(I, v_b)$")
    # ax.set_xticks((0, 1.5, 3, 4.5, 6))
    # ax.set_xticklabels(("$0.0$", "$0.25$", "$0.5$", "$0.75$", "$1.0$"))
    # ax.set_yticks((-2, -1, 0, 1, 2))
    # ax.set_yticklabels(("$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"))

    prettify(ax=ax, legend=True)

    fig.tight_layout()
    fig.savefig(savedir / "value_of_experiments.pdf", dpi=300)
