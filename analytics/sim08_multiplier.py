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

    savedir = Path(__file__).parent / "docs/sim08_multiplier"
    os.makedirs(savedir, exist_ok=True)

    # dist = Uniform(0, 1)
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    # dist = BetaMixture((2, 5), (5, 2), (0.5, 0.5))

    tau = 20  # 0.5
    threshold_indices = np.arange(10, 90).astype(int)
    num_intervals = 100
    prob_state0 = 1

    lam = 1
    beta = 0.95
    mechanism = Mechanism(num_intervals=num_intervals, dist=dist, lam=lam, beta=beta)
    types = mechanism.midpoints

    multipliers = []
    taus = np.linspace(0, 10000, 10)

    for tau in taus:

        # hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
        # hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
        # threshold_index = hyperopt.best()
        threshold_index = 56

        (
            allocations,
            transfers,
            externalities,
            multiplier,
            avg_transfer,
            avg_externality,
            var,
            cvar,
            # y,
            _,
        ) = mechanism.solve(tau, prob_state0, threshold_index, do_print=False)
        multipliers.append(multiplier)

    fig, ax = plt.subplots()
    ax.plot(taus, multipliers)
    fig.savefig(savedir / "multi.pdf", dpi=300)


if __name__ == "__main__":
    main()
