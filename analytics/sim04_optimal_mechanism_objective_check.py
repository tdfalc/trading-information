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
from trading_information.virtual_values_basic import VirtualValues


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

    # tau = 53.84
    # prob_state0 = 1
    ### even if we ensure monotonicty of allocation we need to iron virtual values!!!
    tau = 2  # Above 1 the objective becomes convex, only works with concave functions
    prob_state0 = 1

    # dist = stats.expon()
    dist = BetaMixture((10, 90), (40, 30), (0.5, 0.5))  # bimodal
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))  # bimodal
    # dist = BetaMixture((10, 25), (25, 10), (0.3, 0.7))  # bimodal
    # dist = Uniform(0, 1)  # uniform
    # dist = BetaMixture([80], [20], [1])  # low
    # dist = BetaMixture([20], [80], [1])  # high

    # dist = BetaMixture([2], [7], [1])
    # dist = stats.norm(0, 0.3)

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots()

    allocations = np.linspace(-1, 1, 100)

    midpoint_idx = 50
    print("Midpoint", midpoints[midpoint_idx])

    for i, iron in enumerate((False, True)):
        virtual_values = VirtualValues(
            mechanism.dist,
            midpoints,
            tau,
            prob_state0,
            iron=iron,
        )
        vvs = (
            np.minimum(allocations, 0) * virtual_values.negative[midpoint_idx]
            + np.maximum(allocations, 0) * virtual_values.positive[midpoint_idx]
        )

        ax.plot(allocations, vvs, color=f"C{i}", label=f"iron={iron}")
        best_idx = np.argmax(vvs)
        ax.scatter([allocations[best_idx]], [vvs[best_idx]], color=f"C{i}")

    ax.legend()

    fig.savefig(savedir / "obj.pdf")

    q = -0.5
    pdf = dist.pdf(midpoints)
    pdf_grad = np.gradient(pdf, midpoints)

    fig, ax = plt.subplots()
    ax.plot(midpoints * pdf_grad + 2 * pdf)
    fig.savefig(savedir / "grad.pdf")


if __name__ == "__main__":
    main()
