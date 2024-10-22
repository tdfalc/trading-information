import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex
from tfds.decorators import cache, blind_file_cache
from matplotlib.axes import Axes

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt
from trading_information.distributions import Uniform, BetaMixture, Distribution
from trading_information.virtual_values import VirtualValues
from trading_information.typing import _Floats, _Ints


def add_discontinuities(arr: _Floats, threshold: Optional[float] = None):
    threshold = 1e-3 if threshold is None else threshold
    arr[np.abs(np.diff(arr, prepend=arr[0])) > 0.1] = np.nan
    return arr


def plot_discontinuous_function(
    x: _Floats, y: _Floats, color: str, ax: Optional[Axes] = None, zorder: Optional[int] = 1
):

    if ax is None:
        ax = plt.gca()

    # Mask to filter out NaNs or infinite values
    mask = np.isfinite(y)

    # Split the data into continuous segments
    discontinuities = np.where(np.diff(np.where(mask)[0]) != 1)[0] + 1
    segments_x = np.split(x[mask], discontinuities)
    segments_y = np.split(y[mask], discontinuities)

    # Plot each segment
    for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
        # Plot the continuous segment
        ax.plot(seg_x, seg_y, "-", color=color, zorder=zorder)

        # Highlight the endpoints of the segment
        ax.scatter(
            [seg_x[0], seg_x[-1]],
            [seg_y[0], seg_y[-1]],
            color=color,
            s=30,
            zorder=50,
            facecolor="white",
        )

        # Draw dashed lines between discontinuous segments
        if i < len(segments_x) - 1:
            ax.plot(
                [seg_x[-1], segments_x[i + 1][0]],
                [seg_y[-1], segments_y[i + 1][0]],
                ls="dotted",
                color=color,
                zorder=zorder,
                # alpha=0.5,
                # lw=2,
            )


def main():
    logger = create_logger(__name__)
    logger.info("Running optimal mechanism analysis")

    savedir = Path(__file__).parent / "docs/sim04_optimal_mechanism2"
    os.makedirs(savedir, exist_ok=True)

    experiments = {
        "uniform": {
            "dist": Uniform(0, 1),
            "taus": np.array([0, 1 / 3, 2 / 3, 1, 4 / 3]),
            # "threshold_indices": np.arange(100, 900).astype(int),
            "threshold_indices": np.array([100, 500, 900]).astype(int),
        },
        "bimodal": {
            "dist": BetaMixture((8, 60), (30, 30), (0.5, 0.5)),
            "taus": np.array([0, 1 / 3, 2 / 3, 1, 4 / 3]),
            "threshold_indices": np.arange(400, 600).astype(int),
        },
        "bimodal_2": {
            "dist": BetaMixture((30, 30), (8, 60), (0.5, 0.5)),
            # "taus": np.array([10, 20, 30, 40, 50]),
            # "taus": np.array([0, 1, 5, 10, 50]),
            "taus": np.array([0, 1 / 3, 2 / 3, 1, 4 / 3]),
            "threshold_indices": np.arange(400, 600).astype(int),
        },
    }

    experiments = {
        "uniform": {
            "dist": Uniform(0, 1),
            "taus": np.array([0, 1 / 3, 2 / 3, 1, 4 / 3]),
            # "threshold_indices": np.arange(100, 900).astype(int),
            "threshold_indices": np.array([100, 500, 900]).astype(int),
        },
        # "bimodal": {
        #     "dist": BetaMixture((8, 60), (30, 30), (0.5, 0.5)),
        #     "taus": np.array([0, 1 / 3, 2 / 3, 1, 4 / 3]),
        #     "threshold_indices": np.arange(40, 60).astype(int),
        # },
        # "bimodal": {
        #     "dist": BetaMixture((20, 70), (70, 20), (0.2, 0.8)),
        #     "taus": np.array([0, 1 / 3, 1, 5, 10]),
        #     "threshold_indices": np.arange(40, 60).astype(int),
        # },
        "bimodal_2": {
            "dist": BetaMixture((20, 70), (30, 15), (0.5, 0.5)),
            # "taus": np.array([10, 20, 30, 40, 50]),
            # "taus": np.array([0, 1, 5, 10, 50]),
            # "taus": np.array([10, 20, 50, 100, 200]),
            "taus": np.array([0, 0.2, 1 / 3, 0.6, 2]),
            "threshold_indices": np.arange(400, 600).astype(int),
        },
    }

    num_intervals = 1000
    prob_state0 = 0.0001

    for name, experiment in experiments.items():

        logger.info(f"Running experiment: {name}")

        dist = experiment["dist"]
        threshold_indices = experiment["threshold_indices"]
        taus = experiment["taus"]

        fig, axs = plt.subplots(nrows=4, ncols=len(taus), figsize=(9, 9), sharex=True)
        fig2, axs2 = plt.subplots(nrows=len(taus), ncols=1, figsize=(6, 9))

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist, lam=1, beta=0.95)
        types = mechanism.midpoints

        fig3, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(mechanism.midpoints, dist.pdf(mechanism.midpoints))
        ax.set_ylabel(r"Density")
        ax.set_xlabel("Private Type ($t_i$)")
        prettify(ax=ax)
        fig3.savefig(savedir / f"pdf_{name}.pdf", dpi=300)

        for i, tau in enumerate(taus):

            cache_location = savedir / f"cache"
            os.makedirs(cache_location, exist_ok=True)

            @blind_file_cache(cache_location / f"{name}_tau{tau}.pkl")
            def _run_experiment():

                # Find best threshold index
                hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
                hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
                threshold_index = hyperopt.best()

                # Solve optimal mechanism

                output = mechanism.solve(tau, prob_state0, threshold_index)

                return output, hyperopt

            (
                allocations,
                transfers,
                externalities,
                multiplier,
                avg_transfer,
                avg_externality,
                *_,
            ), hyperopt = _run_experiment()

            print(i, "AVG_TRANSFER", avg_transfer)
            from scipy.integrate import trapezoid

            print(
                "AVG_TRANSFER2",
                trapezoid(transfers * dist.pdf(mechanism.midpoints), x=mechanism.midpoints),
            )
            print(
                "AVG_EXTER2",
                trapezoid(externalities * dist.pdf(mechanism.midpoints), x=mechanism.midpoints),
            )

            # Replace discontinuities with NaN values for plotting
            allocations = add_discontinuities(allocations)
            transfers = add_discontinuities(transfers)
            externalities = add_discontinuities(externalities)

            # Plot hyperopt results
            ax = axs2[i]
            ax.plot(hyperopt.threshold_indices, hyperopt.objectives)
            ax.set_xlabel("Threshold Index")
            ax.set_ylabel("Objective")

            # Plot virtual values
            ax = axs[0, i]
            ax.sharey(axs[0, 0])
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
            ax.axhline(y=multiplier, color="red")
            ax.set_ylabel("Virtual Value" if i == 0 else None)
            prettify(ax=ax)

            # Plot allocations
            ax = axs[1, i]
            ax.sharey(axs[1, 0])
            ax.set_ylabel(r"Allocation ($\xi_j$)" if i == 0 else None)
            plot_discontinuous_function(types, allocations, color="k", ax=ax)
            prettify(ax=ax)

            # Plot transfers
            ax = axs[2, i]
            ax.sharey(axs[2, 0])
            ax.set_ylabel(r"Transfer ($\pi_j$)" if i == 0 else None)
            plot_discontinuous_function(types, transfers, color="k", ax=ax)
            ax.axhline(y=avg_transfer, color="blue", ls="dashed")
            prettify(ax=ax)

            # Plot externalities
            ax = axs[3, i]
            ax.sharey(axs[3, 0])
            ax.set_ylabel(r"Externality" if i == 0 else None)
            ax.set_xlabel("Type ($t_i$)")
            plot_discontinuous_function(types, externalities, color="k", ax=ax)
            ax.axhline(y=avg_externality, color="blue", ls="dashed")
            prettify(ax=ax)

            # print(tau, avg_externality)
            print("--")
            print(" ")

        fig.tight_layout()
        fig.savefig(savedir / f"mechanism_{name}.pdf", dpi=300)

        fig2.tight_layout()
        fig2.savefig(savedir / f"hyperopt_{name}.pdf", dpi=300)


if __name__ == "__main__":
    main()
