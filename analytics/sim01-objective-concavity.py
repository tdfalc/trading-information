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

    savedir = Path(__file__).parent / "docs/sim01_objective_concavity"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 100
    tau = 20
    prob_state0 = 1

    # dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    dist = Uniform(low=0, high=1)  # uniform

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(midpoints, dist.pdf(midpoints))
    ax.set_ylabel(r"Density")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.savefig(savedir / f"densities.pdf", dpi=300)

    threshold_indices = np.arange(1, 99).astype(int)
    multipliers = np.linspace(0, 1, 100)

    # hyperopt_increments = HyperOpt(mechanism, threshold_indices=threshold_indices)
    # hyperopt_increments.run(tau, prob_state0, desc=f"Hyperopt")
    # threshold_index = hyperopt_increments.best()
    # fig, ax = plt.subplots(figsize=(4.5, 3))
    # ax.plot(threshold_indices, hyperopt_increments.objectives)
    # fig.savefig(savedir / f"hyperopt_increments_objectives.pdf", dpi=300)
    # print("threshold_index", threshold_index)

    # hyperopt_virtuals = HyperOptVirtuals(mechanism, multipliers=multipliers)
    # hyperopt_virtuals.run(tau, prob_state0, desc=f"Hyperopt Virtuals")
    # multiplier_best = hyperopt_virtuals.best()
    # fig, ax = plt.subplots(figsize=(4.5, 3))
    # ax.plot(multipliers, hyperopt_virtuals.objectives)
    # fig.savefig(savedir / f"hyperopt_virtuals_objectives.pdf", dpi=300)
    # print("multiplier_best", multiplier_best)

    # allocations, transfers, externalities, multiplier, avg_transfer, avg_externality, obj = (
    #     mechanism.solve_with_increments(tau, prob_state0, threshold_index)
    # )
    # print("avg_transfer", avg_transfer, "avg_externality", avg_externality, "objective", obj)
    # print("multiplier increments", multiplier)

    # (
    #     allocations_vv,
    #     transfers_vv,
    #     externalities_vv,
    #     multiplier_vv,
    #     avg_transfer_vv,
    #     avg_externality_vv,
    #     binaries_vv,
    #     all_virtuals,
    #     binaries2,
    #     obj_vv,
    # ) = mechanism.solve_with_virtuals(
    #     tau,
    #     prob_state0,
    #     multiplier=multiplier_best,
    #     integral_constraint=True,
    # )
    # print(
    #     "avg_transfer_vv",
    #     avg_transfer_vv,
    #     "avg_externality_vv",
    #     avg_externality_vv,
    #     "objective_vv",
    #     obj_vv,
    # )

    # fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    # ax.plot(midpoints, binaries2, color="red")
    # ax2 = ax.twinx()
    # ax2.plot(midpoints, all_virtuals, color="blue")
    # fig.tight_layout()
    # fig.savefig(savedir / f"binaries.pdf", dpi=300)

    # fig, ax = plt.subplots(figsize=(4.5, 3), sharey=False)
    # ax.plot(midpoints, allocations, color="k")
    # ax.plot(midpoints, allocations_vv, color="red")
    # ax.set_ylabel(r"Allocation ($\xi_j$)")
    # ax.set_xlabel("Private Type ($t_i$)")
    # fig.tight_layout()
    # fig.savefig(savedir / f"mechanism.pdf", dpi=300)

    tau = 1
    prob_state0 = 1
    num_intervals = 1000

    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    # dist = Uniform(low=0, high=1)  # uniform

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots(figsize=(4.5, 3))
    allocations = np.linspace(-1, 1, 100)

    positive, negative = VirtualValues.get_ironed_values(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0
    )
    import matplotlib as mpl
    import matplotlib.cm as cm

    # for midpoint in midpoints:
    #     print(
    #         f"Midpoint: {midpoint}, Normalized: {norm(midpoint)}, Color: {m.to_rgba(norm(midpoint))}"
    #     )
    #     ax.plot([0, 1], [midpoint, midpoint], color=m.to_rgba(midpoint))
    from itertools import cycle

    colors = cycle(["green", "blue"])
    ls = cycle(["solid", "dashed"])

    vv = 0

    # idxs = [0, 500, 999]
    # for i in idxs:
    # virtual_values = (
    #     np.minimum(allocations, 0) * negative[i] + np.maximum(allocations, 0) * positive[i]
    # )
    #     ax.plot(allocations, virtual_values, label=i)
    taus = np.linspace(0, 2, 10)

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(taus))
    cmap = cm.get_cmap("viridis")
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for tau in taus:
        i = 0
        positive, negative = VirtualValues.get_ironed_values(
            dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0
        )
        virtual_values = (
            np.minimum(allocations, 0) * negative[i] + np.maximum(allocations, 0) * positive[i]
        )
        ax.plot(allocations, virtual_values, label=i, color=m.to_rgba(tau), zorder=1)

    positive, negative = VirtualValues.get_ironed_values(
        dist=dist, types=midpoints, tau=1, prob_state0=prob_state0
    )
    virtual_values = (
        np.minimum(allocations, 0) * negative[i] + np.maximum(allocations, 0) * positive[i]
    )
    ax.plot(allocations, virtual_values, label=1, color="red", zorder=1, ls="dashed")

    mask = allocations >= 0
    ax.plot(allocations[mask], virtual_values[mask], color="white", ls="solid", zorder=2)
    ax.plot(allocations[mask], virtual_values[mask], color="k", ls="dashed", zorder=3)

    # ax.legend()
    # best_idx = np.argmax(virtual_values)
    # ax.scatter([allocations[best_idx]], [virtual_values[best_idx]])
    fig.savefig(savedir / "concavity.pdf")

    # fig, ax = plt.subplots(figsize=(4.5, 3))
    # virtual_values = VirtualValues(
    #     dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    # )
    # ax.plot(
    #     midpoints,
    #     virtual_values.negative,
    #     color="k",
    #     ls="solid",
    #     label=r"$\phi^{-}$",
    # )

    # ax.plot(
    #     midpoints,
    #     virtual_values.positive,
    #     color="blue",
    #     ls="solid",
    #     label=r"$\phi^{+}$",
    # )
    # ax.legend()

    # virtual_values = VirtualValues(
    #     dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
    # )
    # ax.plot(
    #     midpoints,
    #     virtual_values.negative,
    #     color="k",
    #     ls="dashed",
    #     label=r"$\phi^{-}$",
    # )
    # ax.plot(
    #     midpoints,
    #     virtual_values.positive,
    #     color="blue",
    #     ls="dashed",
    #     label=r"$\phi^{-}$",
    # )
    # # ax2 = ax.twinx()
    # # ax2.plot(midpoints, allocations, color="red")
    # ax.set_ylim(top=10, bottom=-5)

    # ax.set_ylabel("Virtual Value")
    # fig.tight_layout()
    # fig.savefig(savedir / f"virtuals.pdf", dpi=300)


if __name__ == "__main__":
    main()
