import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex
from matplotlib.axes import Axes

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt
from trading_information.distributions import Uniform, BetaMixture
from trading_information.typing import _Floats


def add_discontinuities(arr: _Floats, threshold: Optional[float] = None):
    threshold = 1e-3 if threshold is None else threshold
    arr[np.abs(np.diff(arr, prepend=arr[0])) > 0.1] = np.nan
    return arr


def plot_discontinuous_function(x: _Floats, y: _Floats, color: str, ax: Optional[Axes] = None):

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
        ax.plot(seg_x, seg_y, "-", color=color)

        # Highlight the endpoints of the segment
        ax.scatter(
            [seg_x[0], seg_x[-1]],
            [seg_y[0], seg_y[-1]],
            color=color,
            s=50,
            zorder=5,
            facecolor="white",
        )

        # Draw dashed lines between discontinuous segments
        if i < len(segments_x) - 1:
            ax.plot(
                [seg_x[-1], segments_x[i + 1][0]],
                [seg_y[-1], segments_y[i + 1][0]],
                ls="dashed",
                color=color,
                # alpha=0.5,
                # lw=2,
            )


def main():
    logger = create_logger(__name__)
    logger.info("Running optimal allocations analysis")

    savedir = Path(__file__).parent / "docs/sim04_optimal_mechanism"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000

    tau = 20
    prob_state0 = 1

    print("ALPHA", tau * (1 - 2 * prob_state0))

    dist = Uniform(0, 1)  # uniform
    # dist = stats.expon()
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))  # bimodal
    # dist = BetaMixture([80], [20], [1])  # low
    # dist = BetaMixture([20], [80], [1])  # high

    # dist = BetaMixture([2], [7], [1])
    # dist = stats.norm(0, 0.3)

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(midpoints, dist.pdf(midpoints))
    ax.set_ylabel(r"Density")
    ax.set_xlabel("Private Type ($t_i$)")
    prettify(ax=ax)
    fig.savefig(savedir / f"densities.pdf", dpi=300)

    threshold_indices = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900]).astype(int)

    hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
    hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
    threshold_index = hyperopt.best()
    print("threshold_index", threshold_index)

    # fig, ax = plt.subplots(figsize=(4.5, 3))
    # ax.plot(threshold_types, hyperopt.objectives)
    # prettify(ax=ax)
    # fig.savefig(savedir / f"objectives.pdf", dpi=300)

    # logger.info(f"best threshold type: {threshold_type}")

    allocations, transfers, externalities, multiplier, avg_transfer, avg_externality, obj = (
        mechanism.solve(tau, prob_state0, threshold_index)
    )

    print("multiplier", multiplier * num_intervals)

    print("avg_transfer", avg_transfer, "avg_externality", avg_externality, "objective", obj)

    # allocations = add_discontinuities(allocations)
    # transfers = add_discontinuities(transfers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

    plot_discontinuous_function(midpoints, allocations, color="black", ax=ax1)
    plot_discontinuous_function(midpoints, transfers, color="black", ax=ax2)

    ax1.set_ylabel(r"Allocation ($\xi_j$)")
    ax2.set_ylabel(r"Transfer ($\pi_j$)")

    ax1.set_ylim((-1.1, 1.1))
    ax2.set_ylim((-0.01, 0.26))

    for ax in (ax1, ax2):
        ax.set_xlabel("Private Type ($t_i$)")
        prettify(ax=ax)

    fig.savefig(savedir / f"mechanism.pdf", dpi=300)

    externalities = add_discontinuities(externalities, threshold=0.1)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    plot_discontinuous_function(midpoints, externalities, color="black", ax=ax)
    print(externalities[:10])
    ax.set_ylabel(r"Externality")
    ax.set_xlabel("Private Type ($t_i$)")
    prettify(ax=ax)
    fig.savefig(savedir / f"externalities.pdf", dpi=300)

    from scipy.integrate import trapezoid


if __name__ == "__main__":
    main()
