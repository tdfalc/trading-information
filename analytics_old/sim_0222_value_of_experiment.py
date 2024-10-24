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

    fig, ax2 = plt.subplots(figsize=(6, 3), dpi=300)

    # Value of different experiments to each type
    num_types = 21
    types = np.linspace(0, 1, num_types)
    allocation = np.linspace(-1, 1, num_types)
    exerpiment_values = np.zeros((num_types, num_types))

    for i, informativeness in enumerate(allocation):
        exerpiment_values[i, :] = calculate_exerpiment_value(informativeness, types)

    cmap = LinearSegmentedColormap.from_list("", ["white", "blue"])
    im = ax2.imshow(exerpiment_values[::-1], cmap=cmap, extent=[0, 2, -1, 1])
    # im = ax2.pcolor(exerpiment_values, cmap="jet")
    # im = ax2.pcolor(exerpiment_values, extent=[0, 2, -1, 1])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cmap=cmap, cax=cax)
    cbar.set_label("Value of Experiment ($v_i$)", labelpad=10)
    ax2.set_xlabel("Private Type ($t_i$)")
    ax2.set_ylabel(r"Informativeness ($\xi_j$)")
    # ax2.set_xticks((0, 0.5, 1, 1.5, 2))
    ax2.set_xticklabels(("0.0", "0.25", "0.5", "0.75", "1.0"))
    prettify(ax=ax2)

    fig.tight_layout()
    fig.savefig(savedir / "value.pdf", dpi=300)
