import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

# from tfds.log import create_logger
# from tfds.plotting import prettify, use_tex
from matplotlib.axes import Axes

from trading_information.mechanism_basic import Mechanism
from trading_information.hyperopt import HyperOpt
from trading_information.distributions import Uniform, BetaMixture
from trading_information.typing import _Floats
from trading_information.virtual_values import VirtualValues


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
    # logger = create_logger(__name__)
    # logger.info("Running optimal allocations analysis")

    savedir = Path(__file__).parent / "docs/sim04_optimal_mechanism"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000

    tau = 20
    prob_state0 = 0

    print("ALPHA", tau * (1 - 2 * prob_state0))

    dist = Uniform(0, 1)  # uniform
    # dist = stats.expon()
    # dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))  # bimodal
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
    # prettify(ax=ax)
    fig.savefig(savedir / f"densities.pdf", dpi=300)

    # threshold_indices = np.array([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]).astype(int)

    # hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
    # hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
    # threshold_index = hyperopt.best()
    threshold_index = 500
    print("threshold_index", threshold_index)

    # fig, ax = plt.subplots(figsize=(4.5, 3))
    # ax.plot(threshold_types, hyperopt.objectives)
    # prettify(ax=ax)
    # fig.savefig(savedir / f"objectives.pdf", dpi=300)

    # logger.info(f"best threshold type: {threshold_type}")

    allocations, transfers, externalities, multiplier, avg_transfer, avg_externality, obj = (
        mechanism.solve(tau, prob_state0, threshold_index)
    )
    print("actual multuplier", multiplier)
    multiplier *= -1
    # multiplier /= tau
    multiplier = 10.5

    (
        allocations_vv,
        transfers_vv,
        externalities_vv,
        multiplier_vv,
        avg_transfer_vv,
        avg_externality_vv,
        obj_vv,
    ) = mechanism.solve_virtuals(tau, prob_state0, threshold_index, multi=multiplier)

    # for m in np.linspace(0, 20, 20):
    #     (
    #         *_,
    #         o,
    #     ) = mechanism.solve_virtuals(tau, prob_state0, threshold_index, multi=m)
    #     print(m, o)

    print("multiplier", multiplier)
    print("multiplier_vv", multiplier_vv)

    print("avg_transfer", avg_transfer, "avg_externality", avg_externality, "objective", obj)
    print(
        "avg_transfer_vv",
        avg_transfer_vv,
        "avg_externality_vv",
        avg_externality_vv,
        "objective_vv",
        obj_vv,
    )

    # allocations = add_discontinuities(allocations)
    # transfers = add_discontinuities(transfers)

    fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    ax.plot(midpoints, allocations, color="k")
    ax.set_ylabel(r"Allocation ($\xi_j$)")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.tight_layout()
    fig.savefig(savedir / f"mechanism.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    ax.plot(midpoints, allocations_vv, color="k")
    ax.set_ylabel(r"Allocation ($\xi_j$)")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.tight_layout()
    fig.savefig(savedir / f"mechanism_vv.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    virtuals = VirtualValues(dist, midpoints, tau, prob_state0, iron=False)
    ax.plot(
        midpoints,
        virtuals.negative,
        color="k",
        ls="solid",
        label=r"$\phi^{-}$",
    )
    ax.plot(
        midpoints,
        virtuals.positive,
        color="k",
        ls="dashed",
        label=r"$\phi^{+}$",
    )
    ax.axhline(y=multiplier, color="red")
    ax.set_ylabel("Virtual Value")
    fig.tight_layout()
    fig.savefig(savedir / f"virtuals.pdf", dpi=300)


if __name__ == "__main__":
    main()
