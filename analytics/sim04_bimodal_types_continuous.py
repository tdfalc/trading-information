import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.mechanism import Mechanism
from trading_information.distributions import BetaMixture
from trading_information.hyperopt import HyperOptIncrements
from trading_information.virtual_values import VirtualValues
from trading_information.typing import _Floats
from analytics.sim03_uniform_types_continuous import (
    _plot_informativeness,
    _plot_transfer,
    _plot_externality,
)

logger = create_logger(__name__)


def _plot_virtual_value(
    axs: Axes, midpoints: _Floats, positive: _Floats, negative: _Floats, lag: float, i: int
) -> None:
    ax = axs[0, i]
    ax.sharey(axs[0, 0])
    ax.plot(midpoints, positive, color="k", label=r"$\phi^+$")
    ax.plot(midpoints, negative, color="k", ls="dashed", label=r"$\phi^-$")
    ax.axhline(y=lag, color="red", ls="solid", zorder=0, label=r"$\lambda$")

    where = (negative > lag) & (positive <= lag)
    ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=0.3)
    where = negative <= lag
    ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
    where = positive > lag
    ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
    where = (np.abs(negative - lag) <= 1e-3) | (np.abs(positive - lag) <= 1e-3)
    ax.fill_between(midpoints, negative, positive, where=where, color="red", alpha=0.3)

    prettify(ax=ax, legend=(i == 3), legend_loc="lower right")

    if i == 0:
        ax.set_ylabel("Virtual Value")
        ax.yaxis.set_label_coords(-0.3, 0.5)
    else:
        ax.tick_params(labelleft=False)


def main():
    logger.info("Running bimodal types (continuous) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim04_bimodal_types_continuous"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000
    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))
    prob_state0 = 0
    taus = [0, 0.3, 0.45, 1]

    fig, axs = plt.subplots(4, len(taus), figsize=(6, 7), sharex=True)

    for i, tau in enumerate(taus):

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
        midpoints = mechanism.midpoints

        threshold_indices = np.arange(500, 700)
        hyperopt = HyperOptIncrements(mechanism=mechanism, threshold_indices=threshold_indices)

        hyperopt.run(tau, prob_state0, desc="threshold")
        threshold_index = hyperopt.best()
        print(threshold_index)

        (
            allocations,
            transfers,
            externalities,
            multiplier,
            _,
        ) = mechanism.solve_with_increments(
            tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
        )

        positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

        _plot_virtual_value(axs, midpoints, positive, negative, multiplier, i)
        _plot_informativeness(axs, midpoints, allocations, i)
        _plot_transfer(axs, midpoints, transfers, i)
        _plot_externality(axs, midpoints, externalities, i)

        axs[1, i].axvline(x=0.5)  # remove

        # Add pdf for intuition
        for j in [1, 2, 3]:
            ax = axs[j, i].twinx()
            ax.plot(midpoints, mechanism._pdfs, alpha=0.1)
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(savedir / "bimodal_types_continuous.pdf", dpi=300)


if __name__ == "__main__":
    main()
