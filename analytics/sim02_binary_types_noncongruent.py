from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger


logger = create_logger(__name__)


def tau_critical_low(type_low: float, prob_state0: np.ndarray) -> np.ndarray:
    """Compute the critical τ values for the low type."""
    return type_low / (prob_state0 * (2 * prob_state0 - 1))


def tau_critical_high(type_high: float, prob_state0: np.ndarray) -> np.ndarray:
    """Compute the critical τ values for the high type."""
    return (1 - type_high) / ((1 - 2 * prob_state0) * (1 - prob_state0))


def compute_critical_taus(
    type_low: float, type_high: float, prob_state0s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute valid critical τ values for both low and high types."""
    with np.errstate(divide="ignore", invalid="ignore"):
        taus_low = tau_critical_low(type_low, prob_state0s)
        taus_low[taus_low <= 0] = np.nan

        taus_high = tau_critical_high(type_high, prob_state0s)
        taus_high[taus_high <= 0] = np.nan

    return taus_low, taus_high


def add_arrow_annotation(ax, x: float, label: str):
    """Add an arrow and label to the plot at a given x-coordinate."""
    ax.annotate(
        "",
        xy=(x, -0.05),
        xytext=(x, 0.05),
        arrowprops=dict(arrowstyle="-"),
        annotation_clip=False,
        color="k",
    )
    ax.annotate(label, xy=(x - 0.02, 0), xytext=(x - 0.02, 0.08), annotation_clip=False, color="k")


if __name__ == "__main__":
    logger.info("Running binary types (noncongruent) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim02_binary_types_noncongruent"
    savedir.mkdir(parents=True, exist_ok=True)

    type_high = 0.65
    type_low = 0.1
    num_points = 1000
    prob_state0s = np.linspace(0, 1, num_points)

    taus_low, taus_high = compute_critical_taus(type_low, type_high, prob_state0s)

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(prob_state0s, taus_low, color="k", linestyle="solid", linewidth=1)
    ax.plot(prob_state0s, taus_high, color="k", linestyle="dashed", linewidth=1)

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    add_arrow_annotation(ax, 2 / 3, r"$v_b^h$")
    add_arrow_annotation(ax, 1 / 6, r"$v_b^l$")

    prettify(ax=ax, legend=False, legend_loc="lower left")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$v_s$")
    ax.set_ylabel(r"$\tau$")

    fig.tight_layout()
    fig.savefig(savedir / "binary_types_noncongruent.pdf", dpi=300)
