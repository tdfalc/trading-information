# import os
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# from tfds.plotting import prettify, use_tex
# from tfds.log import create_logger

# from trading_information.mechanism import Mechanism
# from trading_information.distributions import Uniform, Normal, BetaMixture
# from trading_information.virtual_values import VirtualValues

# logger = create_logger(__name__)

# _Floats = np.ndarray[float]


# def add_discontinuities(arr: _Floats, threshold: Optional[float] = 1e-4) -> _Floats:
#     arr_discontinuous = arr.copy()
#     arr_discontinuous[
#         np.abs(np.diff(arr_discontinuous, prepend=arr_discontinuous[0])) > threshold
#     ] = np.nan
#     return arr_discontinuous


# def plot_discontinuous_function(
#     x: _Floats,
#     y: _Floats,
#     color: str,
#     ax: Optional[Axes] = None,
#     zorder: Optional[int] = 1,
# ) -> None:

#     if ax is None:
#         ax = plt.gca()

#     # Filter out NaNs or infinite values
#     mask = np.isfinite(y)

#     # Identify and split the data into continuous segments
#     discontinuities = np.where(np.diff(np.where(mask)[0]) != 1)[0] + 1
#     segments_x = np.split(x[mask], discontinuities)
#     segments_y = np.split(y[mask], discontinuities)

#     # Plot each segment
#     for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
#         ax.plot(seg_x, seg_y, "-", color=color, zorder=zorder, lw=1)
#         ax.scatter(
#             [seg_x[0], seg_x[-1]],
#             [seg_y[0], seg_y[-1]],
#             color=color,
#             s=10,
#             zorder=50,
#             facecolor="w",
#         )

#         # Plot dashed lines between discontinuous segments
#         if i < len(segments_x) - 1:
#             ax.plot(
#                 [seg_x[-1], segments_x[i + 1][0]],
#                 [seg_y[-1], segments_y[i + 1][0]],
#                 ls="dashed",
#                 color=color,
#                 zorder=zorder,
#                 alpha=1,
#                 lw=0.5,
#             )
#             pass


# def _plot_virtual_value(
#     axs: Axes,
#     midpoints: _Floats,
#     positive: _Floats,
#     negative: _Floats,
#     lag: float,
#     i: int,
# ) -> None:
#     ax = axs[0, i]
#     ax.sharey(axs[0, 0])
#     # vv = negative * (midpoints <= 0.5) + positive * (midpoints > 0.5)
#     # vv = add_discontinuities(vv, threshold=0.01)
#     ax.plot(midpoints, positive, color="k", label=r"$\pi^+$", lw=1)
#     ax.plot(midpoints, negative, color="k", ls="dashed", label=r"$\pi^-$", lw=1)
#     # ax.plot(midpoints, vv, color="k", ls="dashed", label=r"$\pi^-$")
#     ax.axhline(y=lag, color="red", ls="solid", zorder=0, label=r"$\lambda$", lw=1)

#     where = (negative > lag) & (positive <= lag)
#     ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=0.3)
#     # where = negative <= lag
#     # ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
#     # where = positive > lag
#     # ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)

#     prettify(ax=ax, legend=(i == 3))

#     if i == 0:
#         ax.set_ylabel("$\pi(v_b)$")
#         ax.yaxis.set_label_coords(-0.4, 0.5)
#     else:
#         ax.tick_params(labelleft=False)


# def _plot_informativeness(axs: Axes, midpoints: _Floats, allocations: _Floats, i: int) -> None:
#     ax = axs[0, i]  # .twinx()
#     # ax.sharey(axs[0, 0].twinx())
#     # ax.axhline(y=0)

#     plot_discontinuous_function(midpoints, add_discontinuities(allocations), "k", ax)
#     prettify(ax=ax, legend=False)
#     if i == 0:
#         ax.set_ylabel("$I (v_b)$")
#         ax.yaxis.set_label_coords(-0.4, 0.5)
#     else:
#         ax.tick_params(labelleft=False)


# def _plot_transfer(axs: Axes, midpoints: _Floats, transfers: _Floats, i: int) -> None:
#     ax = axs[2, i]
#     ax.sharey(axs[2, 0])
#     plot_discontinuous_function(midpoints, add_discontinuities(transfers), "k", ax)
#     prettify(ax=ax, legend=False)
#     if i == 0:
#         ax.set_ylabel("$t (v_b)$")
#         ax.yaxis.set_label_coords(-0.4, 0.5)
#     else:
#         ax.tick_params(labelleft=False)


# def _plot_externality(axs: Axes, midpoints: _Floats, externalities: _Floats, i: int) -> None:
#     ax = axs[3, i]
#     ax.sharey(axs[3, 0])
#     plot_discontinuous_function(midpoints, add_discontinuities(externalities), "k", ax)
#     ax.set_xlabel("$v_b$")
#     prettify(ax=ax, legend=False)
#     if i == 0:
#         ax.set_ylabel(r"$c( I (v_b); \tau)$")
#         ax.yaxis.set_label_coords(-0.4, 0.5)
#     else:
#         ax.tick_params(labelleft=False)


# def main():
#     logger.info("Running uniform types (continuous) analysis")

#     use_tex()

#     savedir = Path(__file__).parent / "docs/sim03_uniform_types_continuous"
#     os.makedirs(savedir, exist_ok=True)

#     num_intervals = 1000
#     threshold_index = int(num_intervals / 2)
#     dist = Uniform(low=0, high=1)

#     # dist = Normal(loc=0.1, scale=1)

#     # dist = BetaMixture(alphas=[1], betas=[0.5], weights=[1])

#     prob_state0 = 1
#     taus = [0, 0.3, 0.7]
#     # taus = [0.5, 0.8, 1, 2]

#     fig, axs = plt.subplots(4, len(taus), figsize=(8, 7), sharex=True)
#     fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
#     fig3, ax3 = plt.subplots(figsize=(2.5, 2.5))
#     fig4, ax4 = plt.subplots(figsize=(2.5, 2.5))
#     ls = ["solid", "dashed", "dashdot"]
#     markers = ["x", "o", "d"]
#     for i, tau in enumerate(taus):

#         mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
#         midpoints = mechanism.midpoints
#         (
#             allocations,
#             transfers,
#             externalities,
#             multiplier,
#             _,
#         ) = mechanism.solve_with_increments(
#             tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
#         )
#         print(tau, multiplier)

#         # positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

#         vvs = VirtualValues(
#             dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
#         )
#         positive, negative = vvs.positive, vvs.negative
#         # positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)
#         if i == 0:
#             ax2.plot(midpoints, positive, lw=0.8, color="k")
#             ax2.plot(midpoints, negative, lw=0.8, color="k")
#             ax5 = ax2.twinx()
#             ax5.plot(midpoints, allocations, lw=0.8, color="k")
#             ax5.plot(midpoints, transfers, lw=0.8, color="k")
#         # ax2.plot(midpoints, transfers, ls=ls[i], color="k", lw=1)

#         _plot_virtual_value(axs, midpoints, positive, negative, multiplier, i)
#         _plot_informativeness(axs, midpoints, allocations, i)
#         _plot_transfer(axs, midpoints, transfers, i)
#         _plot_externality(axs, midpoints, externalities, i)

#     fig.tight_layout()
#     fig.savefig(savedir / "uniform_types_continuous.pdf", dpi=300)

#     ax2.set_xlabel("$v_b$")
#     ax2.set_ylabel("$t(v_b)$")
#     ax3.set_xlabel("$v_b$")
#     ax3.set_ylabel("$t(v_b)$")
#     ax4.set_xlabel("$v_b$")
#     ax4.set_ylabel("$t(v_b)$")
#     prettify(ax=ax2, legend=False)
#     prettify(ax=ax3, legend=False)
#     prettify(ax=ax4, legend=False)

#     fig2.tight_layout()
#     fig2.savefig(savedir / f"uniform_types_continuous_vv.pdf", dpi=300)

#     fig3.tight_layout()
#     fig3.savefig(savedir / f"uniform_types_continuous_all.pdf", dpi=300)

#     fig4.tight_layout()
#     fig4.savefig(savedir / f"uniform_types_continuous_t.pdf", dpi=300)


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
            _,
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

    savedir = Path(__file__).parent / "docs/sim03_uniform_types_continuous"
    os.makedirs(savedir, exist_ok=True)

    #### Objective

    # fig, ax = plt.subplots(figsize=(4.25, 2.5))
    # prob_state0 = 0.1
    # dist = Uniform(low=0, high=1)
    # types = np.linspace(0, 1, 1000)
    # allocations = np.linspace(-1, 1, 1000)
    # label = ["$0$", "$3/5$", "$5/4$", "$2$"]
    # lss = ["solid", "dashed", "dashdot", "dotted"]
    # type_index = 400

    # # i = 400
    # # lss = ["solid", "dotted", "dashed", "dashdot"]
    # # label = ["$0$", "$3/5$", "$5/4$", "$2$"]
    # # for j, tau in enumerate((0, 0.6, 1.3)):
    # colors = ["k", "k", "k"]

    # for n, prob_state0 in enumerate([0.1, 0.5, 0.9]):
    #     fig, ax = plt.subplots(figsize=(2.4, 2.5))
    #     # for i, tau in enumerate((0, 3 / 5, 5 / 4, 2)):
    #     for i, tau in enumerate((0, 1, 2)):

    #         virtuals = VirtualValues(
    #             dist=dist, types=types, tau=tau, prob_state0=prob_state0, iron=True
    #         )

    #         type = types[type_index]
    #         vp = virtuals.positive[type_index]
    #         vn = virtuals.negative[type_index]

    #         # lams = [0.5, 0.1, -0.1, -0.5]
    #         lam = 0  # lams[i]
    #         extent = 10000
    #         ax.plot(
    #             allocations[:extent],
    #             (
    #                 np.minimum(0, allocations) * vn
    #                 + np.maximum(allocations, 0) * vp
    #                 - lam * allocations
    #             )[:extent],
    #             label=r"$\tau=$" + " " + label[i],
    #             color=colors[i],
    #             ls=lss[i],
    #             lw=0.8,
    #         )

    #     # ax.text(-0.5, -0.9, "Reveal $X=1$", size=10, ha="center")
    #     # ax.text(0.5, -0.9, "Reveal $X=0$", size=10, ha="center")
    #     # ax.text(-0.5, -1.2, "Obfuscate $X=0$", size=10, ha="center")
    #     # ax.text(0.5, -1.2, "Obfuscate $X=1$", size=10, ha="center")
    #     # ax.plot([-1, 0], [-0.9, -0.9], "-|", markersize=8, c="k", lw=0.8)
    #     # ax.plot([0, 1], [-0.9, -0.9], "-|", markersize=8, c="k", lw=0.8)

    #     ax.set_xlabel(r"$I(v_b)$")
    #     ax.set_ylabel(r"$I(v_b)\,  \pi (I, v_b)$")
    #     # ax.set_xticks([-1, 0, 1])
    #     ax.set_xticks([-1.0, 0, 1])
    #     ax.set_yticks([-1, 0, 1])
    #     # from matplotlib.ticker import FormatStrFormatter
    #     # import matplotlib.ticker as ticker

    #     # # Define a formatter that retains LaTeX
    #     # def latex_formatter(x, pos):
    #     #     return r"${:.1f}$".format(x)

    #     # ax.xaxis.set_major_formatter(ticker.FuncFormatter(latex_formatter))

    #     ax.set_ylim(bottom=-1, top=1)
    #     ax.set_xlim(left=-1, right=1)
    #     prettify(ax=ax, legend=False if n == 1 else False)
    #     if n == 0:
    #         ax.legend(
    #             facecolor="#eeeeee",
    #             edgecolor="#ffffff",
    #             framealpha=0.5,
    #             handlelength=0.8,
    #             loc="upper left",
    #             labelspacing=0.25,
    #             fontsize=10,
    #         )
    #     # ax.set_title(r"$v_s=$" + " " + f"${prob_state0}$", size=12)
    #     fig.tight_layout()
    #     l = ["a", "b", "c"][n]
    #     fig.savefig(savedir / f"uniform_types_continuous_objective_{l}.pdf", dpi=300)

    ########## multiper

    dist = Uniform(low=0, high=1)
    dist = Normal(loc=0, scale=0.8)
    # dist = Normal(loc=0.5, scale=1)

    fig, ax = plt.subplots(figsize=(4.25, 4.5))

    from tqdm import tqdm

    for prob_state0 in tqdm([0.1, 0.8]):

        taus = np.linspace(0, 4, 50)
        multipliers = []
        max_transfers = []
        from tqdm import tqdm

        for i, tau in enumerate(tqdm(taus)):

            tau_prime = (1 - 2 * prob_state0) ** (-2)
            # print(tau, tau_prime, "thresh")

            mechanism = Mechanism(num_intervals=1000, dist=dist)
            midpoints = mechanism.midpoints
            (
                allocations,
                transfers,
                externalities,
                multiplier,
                _,
            ) = mechanism.solve_with_increments(
                tau=tau, prob_state0=prob_state0, threshold_index=500
            )
            multipliers.append(multiplier)

            vvs = VirtualValues(
                dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
            )
            positive, negative = vvs.positive, vvs.negative

            if tau == 0:
                print("HHHH", positive[500])

            if tau < tau_prime:
                mt = np.median([positive[500], negative[500]])

                #     # mt = 2 * 0.5 + tau * prob_state0 * (1 - 2 * prob_state0)
                max_transfers.append(mt)
            else:
                positive2 = positive
                negative2 = negative
                mt = positive2[500]  # np.median([positive2[500], negative2[500]])

                max_transfers.append(mt)
            # else:
            #     mt = (
            #         1
            #         - 2 * np.max(transfers)
            #         + 2 * 0.5
            #         + tau * (1 - prob_state0) * (1 - 2 * prob_state0)
            #         - 1
            #     )

        multipliers = np.array(multipliers)
        max_transfers = np.array(max_transfers)

        # ax.plot(taus, (multipliers), label=prob_state0)
        ax.plot(taus, max_transfers, label=f"Maxtrans_{prob_state0}")
        ax.plot(taus, multipliers, label=f"multiplier_{prob_state0}")

    prettify(ax=ax, legend=True)

    fig.tight_layout()
    fig.savefig(savedir / "uniform_types_continuous_multipliers.pdf", dpi=300)
