import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.mechanism import Mechanism
from trading_information.distributions import Uniform

logger = create_logger(__name__)


def compute_altruistic_cost(prob_state0: float, tau: float) -> float:
    """Compute the externality-based cost under altruistic allocation."""
    prob_signal1 = 1 - prob_state0 + (1 - 2 * prob_state0) * 0  # No allocation
    return prob_signal1 * (1 - 2 * prob_state0) * tau + tau * prob_state0


def compute_noinfo_cost(prob_state0: float, tau: float, midpoints: np.ndarray) -> float:
    """Compute expected externality under no information sharing."""
    allocations = np.ones_like(midpoints)
    allocations[: len(midpoints) // 2] = -1
    prob_signal1 = (
        1
        - prob_state0
        - prob_state0 * allocations
        + (1 - 2 * prob_state0) * np.minimum(0, -allocations)
    )
    externality = prob_signal1 * (1 - 2 * prob_state0) * tau + tau * prob_state0
    return np.mean(externality)


if __name__ == "__main__":
    logger.info("Running continuous type cost analysis")

    savedir = Path(__file__).parent / "docs/sim06_cost"
    savedir.mkdir(parents=True, exist_ok=True)

    use_tex()

    prob_state0 = 0.9
    dist = Uniform(low=0, high=1)
    taus = np.linspace(0, 2, 50)

    objectives = []
    costs_altruistic = []
    costs_nosharing = []

    for tau in taus:
        mechanism = Mechanism(num_intervals=1000, dist=dist)
        midpoints = mechanism.midpoints

        *_, objective = mechanism.solve_with_increments(
            tau=tau, prob_state0=prob_state0, threshold_index=500
        )
        objectives.append(objective)

        costs_altruistic.append(compute_altruistic_cost(prob_state0, tau))
        costs_nosharing.append(compute_noinfo_cost(prob_state0, tau, midpoints))

    objectives = np.array(objectives)
    costs_altruistic = np.array(costs_altruistic)
    costs_nosharing = np.array(costs_nosharing)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(taus, objectives, lw=0.8, ls="solid", c="k", label="Strategic Versioning")
    ax.plot(taus, -costs_nosharing, lw=0.8, ls="dashdot", c="k", label="No Sharing")
    ax.plot(taus, -costs_altruistic, lw=0.8, ls="dashed", c="k", label="Altruistic")

    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=-0.85)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\mathbb{E}_{\mathcal{V}_b} [ J(I, v_b) ]$")

    prettify(ax=ax, legend=False)

    ax.legend(
        facecolor="#eeeeee",
        edgecolor="#ffffff",
        framealpha=0.5,
        handlelength=0.8,
        loc="upper right",
        labelspacing=0.25,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(savedir / "uniform_types_continuous_costs.pdf", dpi=300)
