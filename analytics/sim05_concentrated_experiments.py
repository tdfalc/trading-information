import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger


logger = create_logger(__name__)


def main():
    logger.info("Running concentrated experiments analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim05_concentrated_experiments"
    os.makedirs(savedir, exist_ok=True)

    num_points = 1000
    xs = np.linspace(-1, 1, num_points)
    upper_bound = np.where(xs <= 0, 1, -2 * xs + 1)
    lower_bound = -xs

    # fig, ax = plt.subplots(figsize=(6, 3))
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # ax.fill_between(xs, upper_bound, lower_bound, colorrk", alpha=0.05)
    ax.fill_between(xs, upper_bound, lower_bound, color="k", alpha=0.05)
    # ax.plot(xs, lower_bound, color="k", ls="dashed", lw=0.8)
    # ax.plot(xs, upper_bound, color="k", ls="dashed", lw=0.8)
    # ax.plot(xs, upper_bound, color="k")
    ax.plot(xs, upper_bound, color="r", ls="dashed", lw=0.8)
    # ax.plot(xs, lower_bound, color="k", ls="solid", alpha=0.1)

    ax.text(-0.35, 0.65, r"$(I_0, I_1)$")
    ax.text(0.55, 0, r"$(I_0^\prime, I_1^\prime)$")
    ax.plot([-0.1, 0.5], [0.6, 0], "-o", markersize=3, color="k", markerfacecolor="k", lw=0.8)

    ax.set_xlabel(r"$I_0 (m) - I_1 (m)$")
    ax.set_ylabel(r"$I_1 (m)$")

    prettify(ax=ax)

    fig.tight_layout()
    fig.savefig(savedir / "boundary.pdf", dpi=300)


if __name__ == "__main__":
    main()
