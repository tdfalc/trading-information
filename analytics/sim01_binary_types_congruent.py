import os
from typing import Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

logger = create_logger(__name__)


def D(prob_state0):
    return (1 - 2 * prob_state0) * (1 - prob_state0)


def tau_critical_low(type_low: float, type_high: float, prob_state0: float, gamma: float) -> float:
    return ((1 - type_low) - gamma * (1 - type_high)) / ((1 - gamma) * D(prob_state0))


def tau_critical_high(type_high: float, prob_state0: float) -> float:
    return (1 - type_high) / D(prob_state0)


def compute_critical_taus(
    type_low: float, type_high: float, prob_state0s: np.ndarray, gamma: float
) -> Tuple:

    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute critical taus for the high type
        taus_critical_high = tau_critical_high(type_high, prob_state0s)
        taus_critical_high[taus_critical_high <= 0] = 1e5

        # Compute critical taus for the low type
        taus_critical_low = tau_critical_low(type_low, type_high, prob_state0s, gamma)
        taus_critical_low[taus_critical_low <= 0] = 1e-5

    return taus_critical_low, taus_critical_high


if __name__ == "__main__":

    logger.info("Running binary types (congruent) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim01_binary_types_congruent"
    os.makedirs(savedir, exist_ok=True)

    type_high = 2 / 3
    type_low = 5 / 6
    num_types = 10000
    prob_state0s = np.linspace(0, 1, num_types)
    gamma = 0.1

    taus_critical_low, taus_critical_high = compute_critical_taus(
        type_low, type_high, prob_state0s, gamma
    )

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(prob_state0s, taus_critical_low, color="k", ls="solid")
    ax.plot(prob_state0s, taus_critical_high, color="k", ls="dashed")

    alpha = 0.5
    ax.fill_between(
        prob_state0s[prob_state0s <= 0.5],
        taus_critical_high[prob_state0s <= 0.5],
        np.nanmax(taus_critical_high),
        facecolor="red",
        alpha=alpha,
        label=r"$(I^l, I^h) = (0, 0)$",
    )

    ax.fill_between(
        prob_state0s,
        taus_critical_low,
        taus_critical_high,
        facecolor="blue",
        alpha=alpha,
        label=r"$(I^l, I^h) = (0, 1)$",
    )

    ax.fill_between(
        prob_state0s,
        0,
        taus_critical_low,
        facecolor="green",
        alpha=alpha,
        label=r"$(I^l, I^h) = (1, 1)$",
    )

    prettify(ax=ax, legend=True)

    ax.set_ylim(top=1, bottom=0.01)
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel(r"Seller's Information ($v_s$)")
    ax.set_ylabel(r"Threshold $\tau$ ($\tau^\star$)")
    fig.tight_layout()
    fig.savefig(savedir / f"binary_types_congruent_gamma{gamma}.pdf", dpi=300)
