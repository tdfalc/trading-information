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
from trading_information.relaxed import Mechanism as MechRelaxed


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

    savedir = Path(__file__).parent / "docs/sim07_cvar"
    os.makedirs(savedir, exist_ok=True)

    # IT seems that symmetric distributions just converge to no information,
    # whereas asymmetric distrivuutions are different..
    dist = Uniform(0, 1)
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    # dist = BetaMixture((2, 5), (5, 2), (0.5, 0.5))

    # AT some point the exteranlit becomes to big and we offer nothing,
    # but for low externality, we can actually offer something for a transfer.
    # dist = BetaMixture((2,), (5,), (1,))

    tau = 1000  # 0.5
    threshold_indices = np.arange(40, 60).astype(int)
    num_intervals = 100
    prob_state0 = 1

    lam = 1
    beta = 0.95
    mechanism = Mechanism(num_intervals=num_intervals, dist=dist, lam=lam, beta=beta)
    types = mechanism.midpoints

    fig, ax = plt.subplots()
    ax.plot(types, dist.pdf(types))
    fig.savefig(savedir / "dist.pdf", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(types, dist.cdf(types))
    ax.axhline(y=0.05)
    fig.savefig(savedir / "cdf.pdf", dpi=300)

    hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
    hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
    threshold_index = hyperopt.best()

    fig, ax = plt.subplots()
    ax.plot(threshold_indices, hyperopt.objectives)
    fig.savefig(savedir / "hypopt2.pdf")
    print("threshold_index", threshold_index)

    threshold_index = 47

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9, 3))

    fig2, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, figsize=(9, 3))

    ax11 = ax1.twinx()
    ax11.plot(mechanism.midpoints, dist.pdf(mechanism.midpoints))

    (
        allocations,
        transfers,
        externalities,
        multiplier,
        *_,
    ) = mechanism.solve(tau, prob_state0, threshold_index, do_print=True)
    alls = allocations
    print("allocations", np.mean(allocations))
    print("MULTUPLIER", multiplier)
    ax1.plot(types, allocations)
    ax_1.plot(types, externalities)
    ax1.set_title("No Thresh")

    ax21 = ax2.twinx()
    ax21.plot(mechanism.midpoints, dist.pdf(mechanism.midpoints))

    (
        allocations,
        transfers,
        externalities,
        multiplier,
        avg_transfer,
        avg_externality,
        var,
        cvar,
        _,
    ) = mechanism.solve_no_infdo(tau, prob_state0, 50, do_print=True)
    print("allocations", np.mean(allocations))
    ax2.plot(types, allocations)
    ax2.set_title("Treshold")
    ax_2.plot(types, externalities)
    fig.savefig(savedir / "results2.pdf")
    fig2.savefig(savedir / "transfers2.pdf")

    fig, ax = plt.subplots()
    virtuals = VirtualValues(dist, types, tau, prob_state0, iron=False)
    ax.plot(
        types,
        virtuals.negative,
        color="green",
        ls="solid",
        label=r"$\phi^{-}$",
    )
    ax.plot(
        types,
        virtuals.positive,
        color="blue",
        ls="solid",
        label=r"$\phi^{+}$",
    )

    virtuals = VirtualValues(dist, types, tau, prob_state0, iron=True)
    ax.plot(
        types,
        virtuals.negative,
        color="green",
        ls="dashed",
        label=r"$\phi^{-}$",
    )
    ax.plot(
        types,
        virtuals.positive,
        color="blue",
        ls="dashed",
        label=r"$\phi^{+}$",
    )
    ax.set_ylim([-2, 2])
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(types, alls, color="red", label="allocation")
    ax2.grid()
    ax.grid(which="minor")

    fig.savefig(savedir / "virtuals.pdf")

    print(alls)

    # # ###### TOIKKKKAAA

    # # # toikka works when tau <=1 because the cost component is weeakly convex...
    # # # about tau=1, it becomes concave

    # theta = 0.1
    # alls = np.linspace(-1, 1, 100)  # why is this externality not linear?
    # # prob_state0 = 0
    # # tau = 1

    # def externality_for_type(type, x):
    #     ps1 = 1 - prob_state0 - prob_state0 * x + (1 - 2 * prob_state0) * np.minimum(0, -x)
    #     externality = ps1
    #     externality *= 1 - 2 * prob_state0
    #     # externality *= prob_state0
    #     externality *= tau

    #     # return externality + tau * prob_state0  # * dist.pdf(type)
    #     return np.maximum(0, x) - x / dist.pdf(type) + (externality + tau * prob_state0)

    # exs = externality_for_type(theta, alls)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(alls, exs)
    # prettify(ax=ax)
    # ax.set_xlabel("Allocations")
    # fig.tight_layout()
    # fig.savefig(savedir / "check.pdf", dpi=200)


if __name__ == "__main__":
    main()
