from pathlib import Path

import numpy as np
import io
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger
from PIL import Image
import matplotlib.pyplot as plt

from sim05_uniform_types_continuous_virtuals import add_discontinuities

logger = create_logger(__name__)


def tau_critical_low(type_low: float, type_high: float, prob_state0: float, phi: float) -> float:
    """Critical τ for the low type when signals are congruent."""
    numerator = (1 - type_low) - phi * (1 - type_high)
    denominator = (1 - phi) * ((1 - 2 * prob_state0) * (1 - prob_state0))
    return numerator / denominator


def tau_critical_high(type_high: float, prob_state0: float) -> float:
    """Critical τ for the high type."""
    return (1 - type_high) / ((1 - 2 * prob_state0) * (1 - prob_state0))


def compute_critical_taus(
    type_low: float, type_high: float, prob_state0s: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute τ thresholds for low and high types over range of beliefs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        taus_high = tau_critical_high(type_high, prob_state0s)
        taus_low = tau_critical_low(type_low, type_high, prob_state0s, phi)
    return taus_low, taus_high


def annotate_vline(ax, x: float, label: str):
    """Add vertical line and label annotation."""
    ax.annotate(
        "",
        xy=(x, -0.05),
        xytext=(x, 0.05),
        arrowprops=dict(arrowstyle="-"),
        annotation_clip=False,
        color="k",
    )
    ax.annotate(label, xy=(x - 0.03, 0), xytext=(x - 0.03, 0.08), annotation_clip=False, color="k")


def plot_critical_thresholds(
    ax: plt.Axes,
    type_low: float,
    type_high: float,
    prob_state0s: np.ndarray,
    phi: float,
    show_xlabel: bool = True,
):
    """Plot τ thresholds for a fixed φ over varying beliefs."""
    taus_low, taus_high = compute_critical_taus(type_low, type_high, prob_state0s, phi)

    ax.plot(prob_state0s, add_discontinuities(taus_low, threshold=0.1), color="k", ls="solid", lw=1)
    ax.plot(
        prob_state0s, add_discontinuities(taus_high, threshold=0.1), color="k", ls="dashed", lw=1
    )

    annotate_vline(ax, 2 / 3, r"$v_b^h$")
    annotate_vline(ax, 5 / 6, r"$v_b^l$")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    if show_xlabel:
        ax.set_xlabel(r"$v_s$")
    ax.set_ylabel(r"$\tau$")
    prettify(ax=ax, legend=False)


if __name__ == "__main__":
    logger.info("Running binary types (congruent) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim01_binary_types_congruent"
    savedir.mkdir(parents=True, exist_ok=True)

    type_high = 2 / 3
    type_low = 5 / 6
    prob_state0s = np.linspace(0, 1, 1000)
    phis = [0.36, 0.5, 0.56]
    labels = ["a", "b", "c"]

    # Save static plot for paper
    for i, phi in enumerate(phis):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        if i == 1:
            ax.axvline(x=0.5, color="k", lw=1)
        plot_critical_thresholds(
            ax,
            type_low=type_low,
            type_high=type_high,
            prob_state0s=prob_state0s,
            phi=phi,
            show_xlabel=True,
        )
        fig.tight_layout()
        fig.savefig(savedir / f"binary_types_congruent_{labels[i]}.pdf", dpi=300)

    # Make gif with pillow
    frames = []
    phis = np.concatenate((np.linspace(0.01, 0.49, 49), np.linspace(0.51, 0.65, 15)))

    for phi in phis:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_critical_thresholds(
            ax,
            type_low=type_low,
            type_high=type_high,
            prob_state0s=prob_state0s,
            phi=phi,
            show_xlabel=True,
        )
        ax.set_title(f"$\\phi = {phi:.2f}$")
        fig.tight_layout()

        # Save as BytesIO object temporarily to avoid saving every frame to disc
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        plt.close(fig)  # Close the figure to free memory

        # Load the image from the buffer
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)

    frames[0].save(
        savedir / "binary_types_congruent.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,  # Duration per frame in milliseconds
        loop=0,  # Loop forever
    )
