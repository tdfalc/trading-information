import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues

import sys

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from tfds.plotting import prettify, use_tex

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from trading_information.typing import _Floats
from typing import Optional

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
from trading_information.typing import _Floats

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm


from tfds.plotting import prettify, use_tex


def add_discontinuities(arr: _Floats, threshold: Optional[float] = None) -> _Floats:
    threshold = 1e-3 if threshold is None else threshold
    threshold = 1e-4
    arr[np.abs(np.diff(arr, prepend=arr[0])) > 0.1] = np.nan
    return arr


def plot_discontinuous_function(
    x: _Floats, y: _Floats, color: str, ax: Optional[Axes] = None, zorder: Optional[int] = 1
) -> None:

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
            s=10,
            zorder=50,
            facecolor="w",
        )

        # Draw dashed lines between discontinuous segments
        if i < len(segments_x) - 1:
            ax.plot(
                [seg_x[-1], segments_x[i + 1][0]],
                [seg_y[-1], segments_y[i + 1][0]],
                ls="dashed",
                color=color,
                zorder=zorder,
                alpha=0.5,
                lw=1,
            )


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
    prettify(ax=ax, legend=(i == 3))
    if i == 0:
        ax.set_ylabel("Virtual Value")
    else:
        ax.tick_params(labelleft=False)


def _plot_informativeness(axs: Axes, midpoints: _Floats, allocations: _Floats, i: int) -> None:
    ax = axs[1, i]
    ax.sharey(axs[1, 0])
    plot_discontinuous_function(midpoints, add_discontinuities(allocations), "k", ax)
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel("Informativeness")
    else:
        ax.tick_params(labelleft=False)


def _plot_transfer(axs: Axes, midpoints: _Floats, transfers: _Floats, i: int) -> None:
    ax = axs[2, i]
    ax.sharey(axs[2, 0])
    plot_discontinuous_function(midpoints, add_discontinuities(transfers), "k", ax)
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel("Transfer")
    else:
        ax.tick_params(labelleft=False)


def _plot_externality(axs: Axes, midpoints: _Floats, externalities: _Floats, i: int) -> None:
    ax = axs[3, i]
    ax.sharey(axs[3, 0])
    plot_discontinuous_function(midpoints, add_discontinuities(externalities), "k", ax)
    ax.set_xlabel("Type ($v_b$)")
    prettify(ax=ax, legend=False)
    if i == 0:
        ax.set_ylabel("Externality")
    else:
        ax.tick_params(labelleft=False)
    print(np.nanmean(externalities))
    print(externalities)
    ax.axhline(y=np.nanmean(externalities), color="red")


def main():
    # logger = create_logger(__name__)
    # logger.info("Running optimal allocations analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim03_uniform_types_continuous"
    os.makedirs(savedir, exist_ok=True)
    # 1.
    num_intervals = 100
    threshold_index = int(num_intervals / 2)
    dist = Uniform(low=0, high=1)
    # prob_state0 = 0.2
    # taus = [0, 1 / 3, 2 / 3, 1]

    # fig, axs = plt.subplots(4, len(taus), figsize=(6, 8), sharex=True)

    # for i, tau in enumerate(taus):

    #     mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    #     midpoints = mechanism.midpoints
    #     (
    #         allocations,
    #         transfers,
    #         externalities,
    #         avg_transfer,
    #         avg_externality,
    #         lag,
    #         obj,
    #     ) = mechanism.solve_with_increments(
    #         tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
    #     )
    #     lag *= num_intervals

    #     positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

    #     _plot_virtual_value(axs, midpoints, positive, negative, lag, i)
    #     _plot_informativeness(axs, midpoints, allocations, i)
    #     _plot_transfer(axs, midpoints, transfers, i)
    #     _plot_externality(axs, midpoints, externalities, i)

    # fig.tight_layout()
    # fig.savefig(savedir / "uniform_types_continuous.pdf", dpi=300)

    # 2.
    prob_state0 = 1
    taus = np.linspace(0, 1, 11)

    fig, ax = plt.subplots(figsize=(7, 3))
    fig2, ax2 = plt.subplots(figsize=(7, 3))

    cmap = plt.cm.viridis
    # create normalization instance
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

    prob_state0s = np.linspace(0, 1, 11)
    for j, tau in enumerate(taus):

        avg_transfers = []
        avg_externalities = []
        objs = []
        for i, prob_state0 in enumerate(prob_state0s):

            mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
            midpoints = mechanism.midpoints
            (
                allocations,
                transfers,
                externalities,
                avg_transfer,
                avg_externality,
                lag,
                obj,
            ) = mechanism.solve_with_increments(
                tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
            )
            lag *= num_intervals
            avg_transfers.append(avg_transfer)
            avg_externalities.append(avg_externality)
            objs.append(obj)

        # fig, ax = plt.subplots()

        ax.plot(prob_state0s, avg_transfers, ls="solid", color=sm.to_rgba(tau))
        ax2.plot(prob_state0s, avg_externalities, color=sm.to_rgba(tau))
    # ax.plot(taus, objs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cbar = fig.colorbar(sm, cmap=cmap, cax=cax)
    cbar.set_label(r"$\tau$", labelpad=10)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cbar = fig.colorbar(sm, cmap=cmap, cax=cax)
    cbar.set_label(r"$\tau$", labelpad=10)

    fig.tight_layout()
    fig.savefig(savedir / "transfers.pdf", dpi=300)
    fig2.tight_layout()
    fig2.savefig(savedir / "externalities.pdf", dpi=300)

    # 3.
    # num_intervals = 100
    # threshold_index = int(num_intervals / 2)
    # dist = Uniform(low=0, high=1)
    # prob_state0 = 0.2
    # prob_state0s = [0, 0.2, 0.5]

    # fig, axs = plt.subplots(4, len(prob_state0s), figsize=(6, 8), sharex=True)

    # tau = 0.7

    # for i, prob_state0 in enumerate(prob_state0s):

    #     mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    #     midpoints = mechanism.midpoints
    #     (
    #         allocations,
    #         transfers,
    #         externalities,
    #         avg_transfer,
    #         avg_externality,
    #         lag,
    #         obj,
    #     ) = mechanism.solve_with_increments(
    #         tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
    #     )
    #     lag *= num_intervals

    #     positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

    #     _plot_virtual_value(axs, midpoints, positive, negative, lag, i)
    #     _plot_informativeness(axs, midpoints, allocations, i)
    #     _plot_transfer(axs, midpoints, transfers, i)
    #     _plot_externality(axs, midpoints, externalities, i)

    # fig.tight_layout()
    # fig.savefig(savedir / "uniform_types_continuous.pdf", dpi=300)


if __name__ == "__main__":
    main()
