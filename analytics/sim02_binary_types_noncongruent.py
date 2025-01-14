import os
from typing import Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

logger = create_logger(__name__)


def tau_critical_low(type_low: float, prob_state0: float) -> float:
    return type_low / (prob_state0 * (2 * prob_state0 - 1))


def tau_critical_high(type_high: float, prob_state0: float) -> float:
    return (1 - type_high) / ((1 - 2 * prob_state0) * (1 - prob_state0))


def compute_critical_taus(type_low: float, type_high: float, prob_state0s: np.ndarray) -> Tuple:

    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute critical taus for the high type
        taus_critical_high = tau_critical_high(type_high, prob_state0s)
        taus_critical_high[taus_critical_high <= 0] = np.nan

        # Compute critical taus for the low type
        taus_critical_low = tau_critical_low(type_low, prob_state0s)
        taus_critical_low[taus_critical_low <= 0] = np.nan

    return taus_critical_low, taus_critical_high


if __name__ == "__main__":

    logger.info("Running binary types (noncongruent) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim02_binary_types_noncongruent"
    os.makedirs(savedir, exist_ok=True)

    type_high = 0.7
    type_low = 0.1
    num_types = 10000
    prob_state0s = np.linspace(0, 1, num_types)

    taus_critical_low, taus_critical_high = compute_critical_taus(type_low, type_high, prob_state0s)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(prob_state0s, taus_critical_low, color="k", ls="solid")
    ax.plot(prob_state0s, taus_critical_high, color="k", ls="dashed")
    alpha = 0.5  # 0.2
    ax.fill_between(
        prob_state0s[prob_state0s <= 0.5],
        taus_critical_high[prob_state0s <= 0.5],
        np.nanmax(taus_critical_high),
        facecolor="blue",
        alpha=alpha,
        label=r"$(I^l, I^h) = (\alpha, 0)$",
    )
    ax.fill_between(
        prob_state0s[prob_state0s >= 0.5],
        taus_critical_low[prob_state0s >= 0.5],
        np.nanmax(taus_critical_low),
        facecolor="red",
        alpha=alpha,
        label=r"$(I^l, I^h) = (0, 1)$",
    )

    ax.fill_between(
        prob_state0s,
        0,
        np.concatenate(
            [taus_critical_high[prob_state0s <= 0.5], taus_critical_low[prob_state0s > 0.5]]
        ),
        facecolor="green",
        alpha=alpha,
        label=r"$(I^l, I^h) = (\alpha, 1)$",
    )

    prettify(ax=ax, legend=True)

    ax.set_ylim(top=1, bottom=0.01)
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel(r"Seller's Information ($v_s$)")
    ax.set_ylabel(r"Threshold $\tau$")

    fig.tight_layout()
    fig.savefig(savedir / "binary_types_noncongruent.pdf", dpi=300)
