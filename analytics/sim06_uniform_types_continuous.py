import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.mechanism import Mechanism
from trading_information.distributions import Uniform, Normal, BetaMixture
from trading_information.virtual_values import VirtualValues

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
        ax.plot(seg_x, seg_y, "-", color=color, zorder=zorder, lw=0.8)
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
                lw=0.8,
            )
            pass


def _plot_virtual_value(
    ax: Axes,
    midpoints: _Floats,
    positive: _Floats,
    negative: _Floats,
    allocations,
    transfers,
    lag: float,
    i: int,
) -> None:

    vv = negative * (allocations < 0) + positive * (allocations > 0)
    vv = negative * (midpoints <= 0.5) + positive * (midpoints > 0.5)
    vv = add_discontinuities(vv, threshold=0.1)

    # ax.plot(midpoints, positive, color="k", label=r"$\pi^+$", lw=0.8, zorder=2)
    # ax.plot(midpoints, negative, color="k", label=r"$\pi^-$", lw=0.8, zorder=2)
    # ax.plot(midpoints, vv, color="k", ls="solid", label=r"$\pi$", lw=0.8)
    plot_discontinuous_function(midpoints, vv, color="k", ax=ax)

    where = (negative > lag) & (positive <= lag)
    # ax.fill_between(midpoints, -10, 10, where=where, color="yellow", alpha=1)
    # ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=1)
    if i <= 1:
        ax.axvline(x=midpoints[np.argmax(where)], color="k", ls="dashdot", lw=0.8)
        ax.axvline(x=midpoints[len(where) - np.argmax(where) - 1], color="k", ls="dashdot", lw=0.8)
    else:
        ax.axvline(x=0.5, color="k", ls="dashdot", lw=0.8)

    ax.set_ylim(top=2, bottom=-2)
    ax.axhline(y=lag, color="red", ls="solid", zorder=1, label=r"$\lambda$", lw=0.8)

    ax.set_ylabel("$\pi(I, v_b)$")
    ax.set_xlabel("$v_b$")

    prettify(ax=ax, legend=False)

    # if i == 0:
    #     ax.set_ylabel("$\pi(v_b)$")
    #     ax.yaxis.set_label_coords(-0.4, 0.5)
    # else:
    #     ax.tick_params(labelleft=False)


def _plot_informativeness(axs: Axes, midpoints: _Floats, allocations: _Floats, i: int) -> None:
    ax = axs[0, i]  # .twinx()
    # ax.sharey(axs[0, 0].twinx())
    # ax.axhline(y=0)

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


def main():
    logger.info("Running uniform types (continuous) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim03_uniform_types_continuous"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000
    threshold_index = int(num_intervals / 2)
    dist = Uniform(low=0, high=1)

    # dist = Normal(loc=0.1, scale=1)

    # dist = BetaMixture(alphas=[1], betas=[0.5], weights=[1])

    prob_state0 = 1
    taus = [0, 0.5, 1]
    # taus = [0.5, 0.8, 1, 2]

    for i, tau in enumerate(taus):

        fig, ax = plt.subplots(figsize=(2.5, 2.5), sharex=True)

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
        midpoints = mechanism.midpoints
        (
            allocations,
            transfers,
            externalities,
            multiplier,
            *_,
        ) = mechanism.solve_with_increments(
            tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
        )
        print(tau, np.max(transfers))

        positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)
        _plot_virtual_value(
            ax, midpoints, positive, negative, allocations, transfers, multiplier, i
        )
        # _plot_informativeness(axs, midpoints, allocations, i)
        # _plot_transfer(axs, midpoints, transfers, i)
        # _plot_externality(axs, midpoints, externalities, i)

        fig.tight_layout()
        l = ["a", "b", "c"][i]
        fig.savefig(savedir / f"uniform_types_continuous_{l}.pdf", dpi=300)


if __name__ == "__main__":
    main()
    use_tex()

    savedir = Path(__file__).parent / "docs/sim06_uniform_types_continuous"
    os.makedirs(savedir, exist_ok=True)

    prob_state0 = 0.9
    dist = Uniform(low=0, high=1)
    # dist = Normal(loc=0, scale=0.9)

    # for j, tau in enumerate((0, 0.6, 1.3)):
    t_prime = (1 - 2 * prob_state0) ** (-2)
    print("t_prime", t_prime)
    for i, tau in enumerate((0, t_prime / 2, t_prime)):

        mechanism = Mechanism(num_intervals=1000, dist=dist)
        midpoints = mechanism.midpoints
        (
            allocations,
            transfers,
            externalities,
            multiplier,
            *_,
        ) = mechanism.solve_with_increments(tau=tau, prob_state0=prob_state0, threshold_index=500)
        print(tau, multiplier)

        vvs = VirtualValues(
            dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
        )
        positive, negative = vvs.positive, vvs.negative

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        lag = multiplier

        vv = negative * (allocations < 0) + positive * (allocations > 0)
        vv = negative * (midpoints <= 0.5) + positive * (midpoints > 0.5)
        vv = add_discontinuities(vv, threshold=0.1)

        # ax.plot(midpoints, positive, color="k", label=r"$\pi^+$", lw=0.8, zorder=2)
        # ax.plot(midpoints, negative, color="k", label=r"$\pi^-$", lw=0.8, zorder=2)
        # ax.plot(midpoints, vv, color="k", ls="solid", label=r"$\pi$", lw=0.8)
        ax.plot(midpoints, negative, color="k", lw=0.8)
        ax.plot(midpoints, positive, color="k", lw=0.8, ls="dashed")

        where = (negative > lag) & (positive <= lag)
        # ax.fill_between(midpoints, -10, 10, where=where, color="yellow", alpha=1)
        # ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=1)
        if i <= 1:
            ax.axvline(
                x=midpoints[np.argmax(where)],
                color="b",
                ls="solid",
                lw=0.8,
                alpha=1,
            )
            ax.axvline(
                x=midpoints[len(where) - np.argmax(where) - 1],
                color="b",
                ls="solid",
                lw=0.8,
                alpha=1,
            )
        else:
            ax.axvline(x=0.5, color="b", ls="solid", lw=0.8, alpha=1)

        s = 50
        if tau < t_prime:
            # ax.scatter(
            #     [0.5],
            #     0.5 * (negative[500] + positive[500]),
            #     zorder=5,
            #     facecolor=None,
            #     color=None,
            #     edgecolor="r",
            #     lw=0.8,
            #     s=s,
            # )
            ax.plot(
                [0.5],
                0.5 * (negative[500] + positive[500]),
                marker="o",
                ms=4,
                markerfacecolor="r",
                markeredgecolor="r",
                markeredgewidth=0.8,
            )
        else:
            ax.plot(
                [0.5],
                [positive[500]],
                marker="o",
                ms=4,
                markerfacecolor="r",
                markeredgecolor="r",
                markeredgewidth=0.8,
            )

        ax.set_ylim(top=2, bottom=-2)
        # ax.set_xlim()
        ax.axhline(y=lag, c="r", lw=0.8)
        # ax.plot(
        #     [-1, 0.5],
        #     [lag, lag],
        #     color="r",
        #     ls="solid",
        #     zorder=1,
        #     label=r"$\lambda$",
        #     lw=0.8,
        #     alpha=1,
        # )
        ax.plot(
            [0.5, 0.5],
            [-3, lag],
            color="r",
            ls="solid",
            zorder=1,
            label=r"$\lambda$",
            lw=0.8,
            alpha=1,
        )

        ax.set_ylabel("$\pi(I, v_b)$")
        ax.set_xlabel("$v_b$")
        ax.set_xlim(left=0, right=1)
        # ax.text(0.5, 0.5, "tet")

        prettify(ax=ax, legend=False)

        letters = ["a", "b", "c"]

        fig.tight_layout()
        fig.savefig(savedir / f"uniform_types_continuous_{letters[i]}.pdf", dpi=300)
