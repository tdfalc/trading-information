import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.distributions import Uniform
from trading_information.virtual_values import VirtualValues

logger = create_logger(__name__)


def plot_virtual_objective(ax, types, allocations, dist, tau_values, prob_state0, type_index):
    """Plot the virtual objective for different tau values at a fixed type."""
    line_styles = ["solid", "dashed", "dashdot", "dotted"]
    labels = ["$0$", "$3/5$", "$5/4$", "$2$"]
    colors = ["k"] * len(tau_values)

    type_value = types[type_index]

    for i, tau in enumerate(tau_values):
        virtuals = VirtualValues(
            dist=dist, types=types, tau=tau, prob_state0=prob_state0, iron=True
        )

        vp = virtuals.positive[type_index]
        vn = virtuals.negative[type_index]

        # Objective function (without constraint multiplier)
        objective = np.minimum(0, allocations) * vn + np.maximum(allocations, 0) * vp

        ax.plot(
            allocations,
            objective,
            label=rf"$\tau=$ {labels[i]}",
            color=colors[i],
            ls=line_styles[i],
            lw=0.8,
        )


if __name__ == "__main__":
    logger.info("Running uniform types continuous (objective) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim04_uniform_types_continuous_objective"
    savedir.mkdir(parents=True, exist_ok=True)

    dist = Uniform(low=0, high=1)
    types = np.linspace(0, 1, 1000)
    allocations = np.linspace(-1, 1, 1000)
    tau_values = (0, 1, 2)
    type_index = 400
    prob_state0_vals = [0.1, 0.5, 0.9]
    panel_labels = ["a", "b", "c"]

    for n, prob_state0 in enumerate(prob_state0_vals):
        fig, ax = plt.subplots(figsize=(2.4, 2.5))

        plot_virtual_objective(ax, types, allocations, dist, tau_values, prob_state0, type_index)

        ax.set_xlabel(r"$I(v_b)$")
        ax.set_ylabel(r"$I(v_b)\,  \pi (I, v_b)$")
        ax.set_xticks([-1.0, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        prettify(ax=ax, legend=False)

        if n == 0:
            ax.legend(
                facecolor="#eeeeee",
                edgecolor="#ffffff",
                framealpha=0.5,
                handlelength=0.8,
                loc="upper left",
                labelspacing=0.25,
                fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(savedir / f"uniform_types_continuous_objective_{panel_labels[n]}.pdf", dpi=300)
