import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.mechanism import Mechanism
from trading_information.distributions import Uniform
from trading_information.virtual_values import VirtualValues

logger = create_logger(__name__)


def add_discontinuities(arr: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Replace large jumps in the array with NaN to create visual breaks."""
    arr_discontinuous = arr.copy()
    jumps = np.abs(np.diff(arr_discontinuous, prepend=arr_discontinuous[0])) > threshold
    arr_discontinuous[jumps] = np.nan
    return arr_discontinuous


def plot_virtual_values(
    ax: plt.Axes,
    midpoints: np.ndarray,
    negative: np.ndarray,
    positive: np.ndarray,
    multiplier: float,
    tau: float,
    t_prime: float,
    index: int,
):
    """Plot positive and negative virtual values with decision boundary and marker."""
    # Plot dashed/solid lines
    ax.plot(midpoints, negative, color="k", lw=0.8)
    ax.plot(midpoints, positive, color="k", lw=0.8, ls="dashed")

    # Plot decision boundary (where switch between neg and pos happens)
    condition = (negative > multiplier) & (positive <= multiplier)
    if index <= 1:
        ax.axvline(midpoints[np.argmax(condition)], color="b", lw=0.8)
        ax.axvline(midpoints[len(condition) - np.argmax(condition) - 1], color="b", lw=0.8)
    else:
        ax.axvline(x=0.5, color="b", lw=0.8)

    # Plot marker at v_b = 0.5
    vb_index = np.searchsorted(midpoints, 0.5)
    if tau < t_prime:
        marker_y = 0.5 * (negative[vb_index] + positive[vb_index])
    else:
        marker_y = positive[vb_index]

    ax.plot(
        [0.5],
        [marker_y],
        marker="o",
        ms=4,
        markerfacecolor="r",
        markeredgecolor="r",
        markeredgewidth=0.8,
    )

    # Horizontal reference line at multiplier
    ax.axhline(y=multiplier, color="r", lw=0.8)

    # Vertical red reference line from v_b=0.5 down to multiplier
    ax.plot(
        [0.5, 0.5],
        [-3, multiplier],
        color="r",
        lw=0.8,
        label=r"$\lambda$",
    )


if __name__ == "__main__":
    logger.info("Running uniform types continuous (virtuals) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim05_uniform_types_continuous_virtuals"
    savedir.mkdir(parents=True, exist_ok=True)

    prob_state0 = 0.9
    dist = Uniform(low=0, high=1)
    t_prime = (1 - 2 * prob_state0) ** -2
    taus = [0, t_prime / 2, t_prime]
    labels = ["a", "b", "c"]

    for i, tau in enumerate(taus):
        mechanism = Mechanism(num_intervals=1000, dist=dist)
        midpoints = mechanism.midpoints

        allocations, transfers, externalities, multiplier, *_ = mechanism.solve_with_increments(
            tau=tau, prob_state0=prob_state0, threshold_index=500
        )

        vvs = VirtualValues(
            dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
        )

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_virtual_values(ax, midpoints, vvs.negative, vvs.positive, multiplier, tau, t_prime, i)

        ax.set_ylim(-2, 2)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$v_b$")
        ax.set_ylabel(r"$\pi(I, v_b)$")

        prettify(ax=ax, legend=False)
        fig.tight_layout()
        fig.savefig(savedir / f"uniform_types_continuous_virtuals_{labels[i]}.pdf", dpi=300)
