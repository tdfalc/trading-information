import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from trading_information.mechanism import Mechanism
from trading_information.distributions import Uniform, Normal, BetaMixture
from trading_information.virtual_values import VirtualValues

logger = create_logger(__name__)

_Floats = np.ndarray[float]


if __name__ == "__main__":

    savedir = Path(__file__).parent / "docs/sim07_cost"
    os.makedirs(savedir, exist_ok=True)

    use_tex()

    prob_state0 = 0.9
    dist = Uniform(low=0, high=1)
    taus = np.linspace(0, 2, 50)
    objectives = []
    costs = []

    buyer_afters = []
    buyer_befores = []
    exs = []
    ts_avg = []
    ts = []
    costs_noinfo = []
    buyers_noinfo = []
    buyer_fullinfo = []
    buyers_mech = []
    buyers_mech_max = []
    buyers_mech_min = []
    for i, tau in enumerate(taus):

        mechanism = Mechanism(num_intervals=1000, dist=dist)
        midpoints = mechanism.midpoints
        (
            allocations,
            transfers,
            externalities,
            multiplier,
            avg_transfer,
            avg_externality,
            obj,
        ) = mechanism.solve_with_increments(tau=tau, prob_state0=prob_state0, threshold_index=500)
        objectives.append(obj)

        def calculate_exerpiment_value(informativeness: float, type: float, transfers) -> float:
            return (
                np.maximum(
                    0,
                    type * informativeness
                    + 1
                    + np.minimum(-informativeness, 0)
                    - (1 - type) * (type <= 0.5)
                    - type * (type > 0.5),
                )
                - transfers
            )

        exs.append(avg_externality)

        all_ts = (
            midpoints * allocations - np.maximum(allocations, 0) - np.cumsum(allocations) / 1000
        )

        # print(avg_transfer, np.mean(all_ts))

        ts.append(np.max(all_ts))
        ts_avg.append(np.mean(all_ts))

        buyers_mech.append(np.mean(calculate_exerpiment_value(allocations, midpoints, transfers)))
        buyers_mech_max.append(
            np.max(calculate_exerpiment_value(allocations, midpoints, transfers))
        )
        buyers_mech_min.append(
            np.min(calculate_exerpiment_value(allocations, midpoints, transfers))
        )
        # print(obj, avg_transfer - avg_externality)

        buyer_fullinfo.append(
            np.mean(calculate_exerpiment_value(np.zeros(len(allocations)), midpoints, 0))
        )

        prob_signal1 = 1 - prob_state0 - prob_state0 * 0 + (1 - 2 * prob_state0) * np.minimum(0, -0)
        fullinfoex = prob_signal1 * (1 - 2 * prob_state0) * tau + tau * prob_state0
        costs.append(fullinfoex)

        allocations_noinfo = np.ones(len(midpoints))
        allocations_noinfo[:500] = -1
        prob_signal1 = (
            1
            - prob_state0
            - prob_state0 * allocations_noinfo
            + (1 - 2 * prob_state0) * np.minimum(0, -allocations_noinfo)
        )
        noinfoex = prob_signal1 * (1 - 2 * prob_state0) * tau + tau * prob_state0
        costs_noinfo.append(np.mean(noinfoex))

        buyers_noinfo.append(np.mean(calculate_exerpiment_value(allocations_noinfo, midpoints, 0)))

    costs = np.array(costs)
    objectives = np.array(objectives)
    buyer_befores = np.array(buyer_befores)
    buyer_afters = np.array(buyer_afters)
    exs = np.array(exs)
    ts = np.array(ts)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(taus, objectives, lw=0.8, ls="solid", c="k", label="Mechanism")
    ax.plot(taus, -costs, lw=0.8, ls="dashed", c="k", label="Altruistic")
    ax.plot(taus, -np.array(costs_noinfo), lw=0.8, ls="dashdot", c="k", label="No Sharing")
    t_prime = (1 - 2 * prob_state0) ** -2
    idx = np.argmax(objectives < 0)
    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(left=0, right=2)
    # ax.set_ylim(bottom=-0.85, top=0.3)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    ax.set_ylabel("$\mathbb{E}_{\mathcal{V}_b} [ J(I, v_b) ]$")
    ax.set_xlabel(r"$\tau$")
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
    fig.savefig(savedir / f"uniform_types_continuous_costs.pdf", dpi=300)

    fig, ax2 = plt.subplots(figsize=(4, 2.5))
    # ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xlim(left=0, right=2)
    # ax2.set_ylim(bottom=-0.85, top=0.3)
    ax2.plot(taus, buyers_mech_max, lw=0.8, ls="solid", c="k")
    ax2.plot(taus, buyers_mech, lw=0.8, ls="solid", c="r")
    ax2.plot(taus, buyer_fullinfo, lw=0.8, ls="dashed", c="k")
    ax2.plot(taus, buyers_noinfo, lw=0.8, ls="dashdot", c="k")
    ax2.set_ylabel("Buyer's Expected Utility$")
    ax2.set_xlabel(r"$\tau$")
    prettify(ax=ax2, legend=False)
    ax2.legend(
        facecolor="#eeeeee",
        edgecolor="#ffffff",
        framealpha=0.5,
        handlelength=0.8,
        loc="upper right",
        labelspacing=0.25,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(savedir / f"uniform_types_continuous_buyer_utility.pdf", dpi=300)
