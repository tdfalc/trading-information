import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence
import os
from pathlib import Path
import sys
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex
from tfds.decorators import cache

from trading_information.mechanism import Mechanism
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues


import matplotlib.cm as cm
import matplotlib as mpl

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d


def main():
    logger = create_logger(__name__)
    logger.info("Running expected externality analysis")

    savedir = Path(__file__).parent / "docs/sim03_virtual_values"
    os.makedirs(savedir, exist_ok=True)

    from scipy import stats

    experiments = {
        "uniform": Uniform(0, 1),
        "gaussian": stats.norm(0, 0.3),
        "expon": stats.expon(),
        "beta_mixture": BetaMixture((8, 60), (30, 30), (0.5, 0.5)),
        "low": BetaMixture([80], [20], [1]),  # low
        "high": BetaMixture([20], [80], [1]),  # low
    }

    # for uniform distribution, when postive virtual value foes above the negative one,
    # we sell no information

    # for beta distribution we get a similar phenomianm except part
    # of the virtual value reminas positiveso we can sell some info

    num_types = 1000
    types = np.linspace(0, 1, num_types)
    prob_state0 = 1
    taus = (0, 0.25, 0.5, 0.9, 1)
    multipliers = (0.5, 0.625, 0.75, 0.95, 1)

    for name, dist in experiments.items():

        fig, axs = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(9, 3))

        for i, ax in enumerate(axs.flatten()):

            tau = taus[i]

            virtuals = VirtualValues(dist, types, tau, prob_state0, iron=False)

            ax.plot(
                types,
                virtuals.negative,
                color="k",
                ls="solid",
                label=r"$\phi^{-}$",
            )
            ax.plot(
                types,
                virtuals.positive,
                color="k",
                ls="dashed",
                label=r"$\phi^{+}$",
            )

            ax.axhline(y=multipliers[i], color="red")

            virtuals = VirtualValues(dist, types, tau, prob_state0, iron=True)

            # ax.plot(
            #     types,
            #     virtuals.negative,
            #     color="red",
            #     ls="solid",
            #     label=None,
            # )
            # ax.plot(
            #     types,
            #     virtuals.positive,
            #     color="red",
            #     ls="dashed",
            #     label=None,
            # )

            if tau == 0:
                title = r"$\alpha=0$"
            elif tau == -1:
                title = r"$\alpha=-1$"
            elif tau == -2:
                title = r"$\alpha=-2$"
            elif tau == 1:
                title = r"$\alpha=1$"
            elif tau == 2:
                title = r"$\alpha=2$"

            # ax.set_ylim([-3, 2.5])

            ax.set_ylabel("Virtual Value")
            ax.set_xlabel("Type ($t_i$)")

            if i == 0:
                prettify(ax=ax, legend=True, legend_loc="upper left")
            else:
                prettify(ax=ax, legend=False)

            ax.set_xticks((0, 0.5, 1))

        fig.savefig(savedir / f"virtual_values_{name}.pdf", dpi=300)


if __name__ == "__main__":

    main()
