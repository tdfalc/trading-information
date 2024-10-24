import os
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.lines import Line2D

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt
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


def state1_probabilities(
    informativeness: float, types: np.ndarray[float], sample_size: int, state: int
):
    prob_signal0_state0 = 1 if informativeness > 0 else 1 + informativeness
    prob_signal1_state1 = 1 if informativeness <= 0 else 1 - informativeness

    prob_signal0_state1 = 1 - prob_signal1_state1
    prob_signal1_state0 = 1 - prob_signal0_state0

    prob_signal0 = prob_signal0_state0 * types + prob_signal0_state1 * (1 - types)
    prob_signal1 = prob_signal1_state0 * types + prob_signal1_state1 * (1 - types)

    prob_state0_signal0 = prob_signal0_state0 * types / prob_signal0
    prob_state0_signal1 = prob_signal1_state0 * types / prob_signal1

    signals = np.random.uniform(0, 1, size=(sample_size, len(types))) <= (
        prob_signal1_state1 if state == 1 else prob_signal1_state0
    )

    probs_state0 = signals * prob_state0_signal1 + (1 - signals) * prob_state0_signal0

    return 1 - probs_state0.mean(axis=0)


if __name__ == "__main__":

    use_tex()

    savedir = Path(__file__).parent / "docs/sim02_value_of_experiment"
    os.makedirs(savedir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 3), width_ratios=[2, 1], height_ratios=[1], dpi=300
    )

    # Value of different experiments to each type
    num_types = 21
    types = np.linspace(0, 1, num_types)
    allocation = np.linspace(-1, 1, num_types)
    exerpiment_values = np.zeros((num_types, num_types))

    for i, informativeness in enumerate(allocation):
        exerpiment_values[i, :] = calculate_exerpiment_value(informativeness, types)

    cmap = LinearSegmentedColormap.from_list("", ["white", "blue"])
    im = ax2.imshow(exerpiment_values[::-1], cmap=cmap, extent=[0, 2, -1, 1])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cmap=cmap, cax=cax)
    cbar.set_label("Value of Experiment ($v_i$)", labelpad=10)
    ax2.set_xlabel("Private Type ($t_i$)")
    ax2.set_ylabel(r"Informativeness ($\xi_j$)")
    ax2.set_xticks((0, 0.5, 1, 1.5, 2))
    # ax2.set_xticklabels((0.0, 0.25, 0.5, 0.75, 1.0))
    prettify(ax=ax2)

    # Dsitribution of actions for fully informative and uninformative experiments
    sample_size = 10000
    offset = 0.3
    facecolor = ("w", 0)
    num_types = 10
    types = np.linspace(0.01, 0.99, num_types)
    # colors = get_colors()
    scatter_scale = 1000
    xs = np.zeros(len(types))
    p = (0, (5, 1))
    line_styles = [p, "solid", "dotted"]
    state = 0
    colors = ["black", "magenta"]

    handles = []

    for i, type in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]):

        probs_state1 = state1_probabilities(
            informativeness=-0.999,
            types=np.array([type]),
            sample_size=sample_size,
            state=state,
        )

        off = 0.01
        ax1.bar(x=type - off, height=probs_state1[0], width=0.02, color="magenta", edgecolor="k")
        ax1.bar(
            x=type + off, height=1 - probs_state1[0], width=0.02, color="limegreen", edgecolor="k"
        )

        # ax1.bar(x=type, height=1, width=0.1, color="limegreen", edgecolor="k")
        # ax1.bar(x=type, height=probs_state1[0], width=0.1, color="magenta", edgecolor="k")

        print(probs_state1)

        # ax2.scatter(
        #     types,
        #     xs + (0.00 if i == 1 else 0),
        #     s=scatter_scale * (1 - probs_state1),
        #     edgecolor=edgecolor,
        #     facecolor=facecolor,
        #     ls=ls,
        #     lw=1.5,
        #     zorder=1 - i,
        # )

    ax1.set_ylabel("P($a_i$ = 1)")
    ax1.set_xlabel("Private Type ($t_i$)")
    prettify(ax=ax1)

    fig.tight_layout()
    fig.savefig(savedir / "value.pdf", dpi=300)
