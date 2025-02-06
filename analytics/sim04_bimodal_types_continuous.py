# import os
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# from tfds.plotting import prettify, use_tex
# from tfds.log import create_logger

# from trading_information.mechanism import Mechanism
# from trading_information.distributions import BetaMixture
# from trading_information.hyperopt import HyperOptIncrements
# from trading_information.virtual_values import VirtualValues
# from analytics.sim03_uniform_types_continuous import (
#     _plot_informativeness,
#     _plot_transfer,
#     _plot_externality,
# )

# logger = create_logger(__name__)

# _Floats = np.ndarray[float]


# def _plot_virtual_value(
#     axs: Axes, midpoints: _Floats, positive: _Floats, negative: _Floats, lag: float, i: int
# ) -> None:
#     ax = axs[0, i]
#     ax.sharey(axs[0, 0])
#     ax.plot(midpoints, positive, color="k", label=r"$\phi^+$")
#     ax.plot(midpoints, negative, color="k", ls="dashed", label=r"$\phi^-$")
#     ax.axhline(y=lag, color="red", ls="solid", zorder=0, label=r"$\lambda$")

#     where = (negative > lag) & (positive <= lag)
#     ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=0.3)
#     where = negative <= lag
#     ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
#     where = positive > lag
#     ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
#     where = (np.abs(negative - lag) <= 1e-3) | (np.abs(positive - lag) <= 1e-3)
#     ax.fill_between(midpoints, negative, positive, where=where, color="red", alpha=0.3)

#     prettify(ax=ax, legend=(i == 3), legend_loc="lower right")

#     if i == 0:
#         ax.set_ylabel("Virtual Value")
#         ax.yaxis.set_label_coords(-0.3, 0.5)
#     else:
#         ax.tick_params(labelleft=False)


# def main():
#     logger.info("Running bimodal types (continuous) analysis")

#     use_tex()

#     savedir = Path(__file__).parent / "docs/sim04_bimodal_types_continuous"
#     os.makedirs(savedir, exist_ok=True)

#     num_intervals = 100
#     dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))
#     prob_state0 = 0
#     taus = [0, 0.3, 0.45, 1]

#     prob_state0 = 1
#     taus = [0, 1, 4, 60]

#     fig, axs = plt.subplots(4, len(taus), figsize=(6, 7), sharex=True)

#     for i, tau in enumerate(taus):

#         mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
#         midpoints = mechanism.midpoints

#         threshold_indices = np.arange(1, 99)
#         hyperopt = HyperOptIncrements(mechanism=mechanism, threshold_indices=threshold_indices)

#         hyperopt.run(tau, prob_state0, desc="threshold")
#         threshold_index = hyperopt.best()
#         print(threshold_index)

#         (
#             allocations,
#             transfers,
#             externalities,
#             multiplier,
#             _,
#         ) = mechanism.solve_with_increments(
#             tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
#         )

#         positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

#         _plot_virtual_value(axs, midpoints, positive, negative, multiplier, i)
#         _plot_informativeness(axs, midpoints, allocations, i)
#         _plot_transfer(axs, midpoints, transfers, i)
#         _plot_externality(axs, midpoints, externalities, i)

#         axs[1, i].axvline(x=0.5)  # remove

#         # Add pdf for intuition
#         for j in [1, 2, 3]:
#             ax = axs[j, i].twinx()
#             ax.plot(midpoints, mechanism._pdfs, alpha=0.1)
#             ax.set_yticks([])

#     fig.tight_layout()
#     fig.savefig(savedir / "bimodal_types_continuous.pdf", dpi=300)


# if __name__ == "__main__":
#     main()


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
from analytics.sim03_uniform_types_continuous import (
    _plot_informativeness,
    _plot_transfer,
    _plot_externality,
)

logger = create_logger(__name__)

_Floats = np.ndarray[float]


def add_discontinuities(arr: _Floats, threshold: Optional[float] = 1e-4) -> _Floats:
    arr_discontinuous = arr.copy()
    arr_discontinuous[
        np.abs(np.diff(arr_discontinuous, prepend=arr_discontinuous[0])) > threshold
    ] = np.nan
    return arr_discontinuous


def plot_discontinuous_function(
    x: _Floats,
    y: _Floats,
    color: str,
    ax: Optional[Axes] = None,
    zorder: Optional[int] = 1,
) -> None:

    if ax is None:
        ax = plt.gca()

    # Filter out NaNs or infinite values
    mask = np.isfinite(y)

    # Identify and split the data into continuous segments
    discontinuities = np.where(np.diff(np.where(mask)[0]) != 1)[0] + 1
    segments_x = np.split(x[mask], discontinuities)
    segments_y = np.split(y[mask], discontinuities)

    # Plot each segment
    for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
        ax.plot(seg_x, seg_y, "-", color=color, zorder=zorder, lw=1)
        ax.scatter(
            [seg_x[0], seg_x[-1]],
            [seg_y[0], seg_y[-1]],
            color=color,
            s=10,
            zorder=50,
            facecolor="w",
        )

        # Plot dashed lines between discontinuous segments
        if i < len(segments_x) - 1:
            ax.plot(
                [seg_x[-1], segments_x[i + 1][0]],
                [seg_y[-1], segments_y[i + 1][0]],
                ls="dashed",
                color=color,
                zorder=zorder,
                alpha=1,
                lw=0.5,
            )


def _plot_informativeness(axs: Axes, midpoints: _Floats, allocations: _Floats, i: int) -> None:
    ax = axs[1, i]
    ax.sharey(axs[1, 0])

    plot_discontinuous_function(midpoints, add_discontinuities(allocations), "k", ax)
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel("$I (v_b)$")
        ax.yaxis.set_label_coords(-0.4, 0.5)
    else:
        ax.tick_params(labelleft=False)


def _plot_transfer(axs: Axes, midpoints: _Floats, transfers: _Floats, i: int) -> None:
    ax = axs[2, i]
    ax.sharey(axs[2, 0])
    plot_discontinuous_function(midpoints, add_discontinuities(transfers), "k", ax)
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel("$t (v_b)$")
        ax.yaxis.set_label_coords(-0.4, 0.5)
    else:
        ax.tick_params(labelleft=False)


def _plot_externality(axs: Axes, midpoints: _Floats, externalities: _Floats, i: int) -> None:
    ax = axs[3, i]
    ax.sharey(axs[3, 0])
    plot_discontinuous_function(midpoints, add_discontinuities(externalities), "k", ax)
    ax.set_xlabel("$v_b$")
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel(r"$c( I (v_b); \tau)$")
        ax.yaxis.set_label_coords(-0.4, 0.5)
    else:
        ax.tick_params(labelleft=False)


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

    num_intervals = 100
    from trading_information.distributions import Uniform

    dist = Uniform(low=0, high=1)

    multipliers = []
    from tqdm import tqdm

    allocations500 = []
    prob_state0 = 0.1
    taus = [0, 0.6, 1.3, 5]
    fig, axs = plt.subplots(4, len(taus), figsize=(6, 7), sharex=True)

    for i, tau in enumerate(tqdm(taus)):

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
        midpoints = mechanism.midpoints

        threshold_index = 50

        (
            allocations,
            transfers,
            externalities,
            multiplier,
            obj,
        ) = mechanism.solve_with_increments(
            tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
        )
        multipliers.append(multiplier)

        # allocations500.append(allocations[500])

        positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)
        print(prob_state0, "tau", tau, "obj", obj, np.min(positive))

        _plot_virtual_value(axs, midpoints, positive, negative, multiplier, i)
        _plot_informativeness(axs, midpoints, allocations, i)
        _plot_transfer(axs, midpoints, transfers, i)
        _plot_externality(axs, midpoints, externalities, i)

        # axs[1, i].axvline(x=0.5)  # remove

        # Add pdf for intuition
        for j in [1, 2, 3]:
            ax = axs[j, i].twinx()
            ax.plot(midpoints, mechanism._pdfs, alpha=0.1)
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(savedir / "bimodal_types_continuous1.pdf", dpi=300)

    # fig, ax = plt.subplots()
    # ax.plot(taus, multipliers)
    # ax2 = ax.twinx()
    # ax2.plot(taus, allocations500)

    # fig.tight_layout()
    # fig.savefig(savedir / "bimodal_types_continuous.pdf", dpi=300)


if __name__ == "__main__":
    main()
