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

    experiments = {
        "uniform": Uniform(0, 1),
        "beta_mixture": BetaMixture((8, 60), (30, 30), (0.5, 0.5)),
    }

    num_types = 1000
    types = np.linspace(0, 1, num_types)
    alphas = (0, 1, 2, -1, -2)

    for name, dist in experiments.items():

        fig, axs = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(9, 3))

        for i, ax in enumerate(axs.flatten()):

            alpha = alphas[i]

            virtuals = VirtualValues(dist, types, alpha, iron=False)

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

            # virtuals = VirtualValues(dist, types, alpha, iron=True)

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

            if alpha == 0:
                title = r"$\alpha=0$"
            elif alpha == -1:
                title = r"$\alpha=-1$"
            elif alpha == -2:
                title = r"$\alpha=-2$"
            elif alpha == 1:
                title = r"$\alpha=1$"
            elif alpha == 2:
                title = r"$\alpha=2$"

            ax.set_ylim([-3, 2.5])

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
