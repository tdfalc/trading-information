import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues


def main():
    # logger = create_logger(__name__)
    # logger.info("Running optimal allocations analysis")

    savedir = Path(__file__).parent / "docs/sim03_optimal_mechanism"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 100
    taus = np.linspace(0, 1, 3)
    taus = [0.1, 0.15, 0.17, 0.19]
    taus = [0, 0.2, 100]
    allocations_20 = np.zeros(len(taus))

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i, tau in enumerate(taus):
        prob_state0 = 1

        dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
        # dist = Uniform(low=0, high=1)  # uniform

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
        midpoints = mechanism.midpoints

        if tau > 1000:
            allocations = np.ones(len(midpoints))
            allocations[: int(len(midpoints) / 2)] = -1
            tau = 0.99
            lag = None
        else:

            tau_opt = tau
            threshold_indices = np.arange(30, 70)
            hyperopt = HyperOpt(mechanism=mechanism, threshold_indices=threshold_indices)
            hyperopt.run(tau_opt, prob_state0, desc="threshold")
            threshold_index = hyperopt.best()
            allocations, bins, avg_transfer, avg_externality, lag, obj = (
                mechanism.solve_with_increments(
                    tau=tau_opt,
                    prob_state0=prob_state0,
                    threshold_index=threshold_index,
                    doprint=True,
                )
            )

        print(tau, "avg_transfer", avg_transfer, "avg_externality", avg_externality, "obj", obj)

        transfers = mechanism._allocations_to_transfers(allocations)
        externalities = mechanism._allocations_to_externalities(allocations, tau, prob_state0)

        weighted_pdf = 1  # mechanism._pdfs / np.sum(mechanism._pdfs)

        ax.plot(midpoints, allocations, label=f"{tau:2g}")
        ax1.plot(midpoints, transfers * weighted_pdf, label=f"{tau:2g}")
        ax2.plot(midpoints, externalities * weighted_pdf, label=f"{tau:2g}", ls="solid")

        fig3, ax3 = plt.subplots(figsize=(4.5, 3))
        virtual_values = VirtualValues(
            dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
        )
        ax3.plot(
            midpoints,
            virtual_values.negative,
            color="k",
            ls="solid",
            label=r"$\phi^{-}$",
        )

        ax3.plot(
            midpoints,
            virtual_values.positive,
            color="blue",
            ls="solid",
            label=r"$\phi^{+}$",
        )
        if lag is not None:
            ax3.axhline(y=lag * 100, color="red")
        ax3.legend()
        ax3.set_ylabel("Virtual Value")
        ax3.set_ylim(bottom=-0.05)
        fig3.tight_layout()
        fig3.savefig(savedir / f"virtuals_{tau}.pdf", dpi=300)

    ax.legend()
    ax.axhline(y=0, c="k", zorder=0)
    ax1.legend()
    ax2.legend()

    # ax2.axhline(y=0.175, c="k", zorder=0)

    fig.savefig(savedir / "taus.pdf")
    fig1.savefig(savedir / "tr.pdf")
    fig2.savefig(savedir / "ex.pdf")


if __name__ == "__main__":
    main()
