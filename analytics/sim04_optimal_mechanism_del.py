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

    num_intervals = 100

    # dist = stats.expon()
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))  # bimodal

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    tau = 2
    prob_state0 = 1
    # threshold_indices = np.arange(30, 40).astype(int)
    # hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
    # hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
    # threshold_index = hyperopt.best()
    threshold_index = 37
    print("threshold_index", threshold_index)
    allocations, transfers, externalities, multiplier, avg_transfer, avg_externality, obj = (
        mechanism.solve(tau, prob_state0, threshold_index=threshold_index)
    )
    print("avg_transfer", avg_transfer, "avg_externality", avg_externality, "objective", obj)
    print("actual multuplier", multiplier)

    (
        allocations_vv,
        transfers_vv,
        externalities_vv,
        multiplier_vv,
        avg_transfer_vv,
        avg_externality_vv,
        obj_vv,
    ) = mechanism.solve(tau, prob_state0, threshold_index=50)
    print(
        "avg_transfer_vv",
        avg_transfer_vv,
        "avg_externality_vv",
        avg_externality_vv,
        "objective_vv",
        obj_vv,
    )
    print("actual multuplier_vv", multiplier_vv)

    virtual_values = VirtualValues(mechanism.dist, mechanism.midpoints, tau, prob_state0, iron=True)

    (
        allocations_vv2,
        transfers_vv2,
        externalities_vv2,
        multiplier_vv2,
        avg_transfer_vv2,
        avg_externality_vv2,
        obj_vv2,
        all_virtuals,
        binaries2,
    ) = mechanism.solve_virtuals(
        tau, prob_state0, threshold_index, multi=0.076, virtual_values=virtual_values
    )
    print(
        "avg_transfer_vv2",
        avg_transfer_vv2,
        "avg_externality_vv2",
        avg_externality_vv2,
        "objective_vv2",
        obj_vv2,
    )
    print("actual multuplier_vv2", avg_externality_vv2)

    fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    ax.plot(midpoints, allocations, color="k")
    ax.plot(midpoints, allocations_vv, color="purple")
    ax.plot(midpoints, allocations_vv2, color="red")
    ax.set_ylabel(r"Allocation ($\xi_j$)")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.tight_layout()
    fig.savefig(savedir / f"mechanism.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    ax.plot(midpoints, transfers, color="k")
    ax.plot(midpoints, transfers_vv, color="purple")
    ax.set_ylabel(r"Allocation ($\xi_j$)")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.tight_layout()
    fig.savefig(savedir / f"transfers.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    ax.plot(midpoints, externalities, color="k")
    ax.plot(midpoints, externalities_vv, color="purple")
    ax.set_ylabel(r"Allocation ($\xi_j$)")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.tight_layout()
    fig.savefig(savedir / f"externalities.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.plot(
        midpoints,
        virtual_values.negative,
        color="blue",
        ls="solid",
        label=r"$\phi^{-}$",
    )
    ax.plot(
        midpoints,
        virtual_values.positive,
        color="blue",
        ls="dashed",
        label=r"$\phi^{+}$",
    )
    ax.axhline(y=multiplier, color="limegreen")
    ax2 = ax.twinx()
    ax2.plot(midpoints, allocations, color="k")
    ax2.plot(midpoints, allocations_vv, color="purple")
    ax2.plot(midpoints, allocations_vv2, color="red")

    print("-")
    print(np.sum(allocations))
    print(np.sum(allocations_vv))
    print(np.sum(allocations_vv2))
    # ax.axhline(y=0, color="gray")

    ax.set_ylabel("Virtual Value")
    fig.tight_layout()
    fig.savefig(savedir / f"virtuals.pdf", dpi=300)

    ms = []
    objs = []
    from tqdm import tqdm

    # for m in tqdm(np.linspace(0.01, 0.09, 100)):
    #     (
    #         *_,
    #         o,
    #         _,
    #         _,
    #     ) = mechanism.solve_virtuals(
    #         tau, prob_state0, threshold_index, multi=m, virtual_values=virtual_values
    #     )
    #     ms.append(m)
    #     objs.append(o)
    #     print(m, o)

    # print(ms[np.argmin(objs)])


if __name__ == "__main__":
    main()
