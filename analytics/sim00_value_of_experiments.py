import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

logger = create_logger(__name__)


def calculate_exerpiment_value(informativeness: float, type: float) -> float:
    return np.maximum(
        0, 1 - informativeness * ((informativeness >= 0) - type) - np.maximum(type, 1 - type)
    )


if __name__ == "__main__":

    logger.info("Running value of experiments analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim00_value_of_experiments"
    os.makedirs(savedir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 2.5))

    num_types = 1000
    types = np.linspace(0, 1, num_types)
    allocations = [-0.8, 0, 0.36]
    labels = [r"$-4/5$", r"0", r"$2/5$"]
    linestyles = ["dashdot", "solid", "dashed"]
    experiment_values = np.zeros((num_types, num_types))

    for i, allocation in enumerate(allocations):
        experiment_values = calculate_exerpiment_value(allocation, types)
        ax.plot(types, experiment_values, color="k", lw=1, label=labels[i], ls=linestyles[i])

    ax.set_xlabel("$v_b$")
    ax.set_ylabel(r"$\delta(I, v_b)$")
    prettify(ax=ax, legend=True)
    fig.tight_layout()
    fig.savefig(savedir / "value_of_experiments.pdf", dpi=300)
