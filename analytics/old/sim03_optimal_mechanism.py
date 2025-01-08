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
    tau = 0.99
    prob_state0 = 1

    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    # dist = Uniform(low=0, high=1)  # uniform

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    multipliers = np.linspace(0.668, 0.669, 1000)  # np.linspace(0, 1, 100)
    multipliers = np.linspace(-1, 1, 100)
    # multipliers = [-0.43274917757547826]
    # # multipliers = [0.6683783783783784]
    # # multipliers = [0.6682082082082083]  # when tau = 0
    # # multipliers = [0.9242042042042042]
    threshold_indices = np.arange(20, 70)

    hyperopt = HyperOpt(mechanism=mechanism, threshold_indices=threshold_indices)
    hyperopt_virtuals = HyperOptVirtuals(mechanism=mechanism, multipliers=multipliers)

    hyperopt.run(tau, prob_state0, desc="threshold")
    threshold_index = hyperopt.best()

    fig, ax = plt.subplots()
    ax.plot(threshold_indices, hyperopt._objectives)
    fig.savefig(savedir / "thresholds.pdf")
    print("threshold_index", threshold_index)

    hyperopt_virtuals.run(tau, prob_state0, desc="multipliers")
    multiplier = hyperopt_virtuals.best()
    # ultiplier = 0.15239509107604077
    print("multiplier", multiplier)

    # fig, ax = plt.subplots()
    # ax.plot(multipliers, hyperopt_virtuals._objectives)
    # fig.savefig(savedir / "multipliers.pdf")
    # print("multiplier", multiplier)

    # multiplier = 0.6682059652693417
    # multiplier = 0.6682056655
    # threshold_index = 66  # 645

    allocations, bins, avg_transfer, avg_externality, lag, obj = mechanism.solve_with_increments(
        tau=tau, prob_state0=prob_state0, threshold_index=threshold_index, doprint=True
    )
    fig, ax = plt.subplots()
    ax.plot(midpoints, allocations, label="increments")
    print(
        "avg_transfer",
        avg_transfer,
        "avg_externality",
        avg_externality,
        "objective",
        avg_transfer - avg_externality,
        "obj",
        obj,
    )

    allocations_virtuals, bins, avg_transfer, avg_externality, obj = mechanism.solve_with_virtuals(
        tau=tau,
        prob_state0=prob_state0,
        multiplier=multiplier,
        integral_constraint=False,
        pooling=False,
        # set_allocations=allocations,
    )
    # ax.plot(midpoints, allocations_virtuals, label="virtuals_int", color="green")
    print(
        "avg_transfer",
        avg_transfer,
        "avg_externality",
        avg_externality,
        "objective",
        avg_transfer - avg_externality,
        "obj",
        obj,
    )

    print("sum", np.sum(allocations_virtuals))

    # # allocations_virtuals.tofile("./test2.dat")
    # np.save("test3.npy", allocations_virtuals)

    virtual_values = VirtualValues(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    )
    neg = virtual_values.negative
    pos = virtual_values.positive

    print("neg", neg[40], "pos", pos[40], (neg[40] + pos[40]) / 2)

    allocations_try = np.zeros(len(allocations))
    # for i in range(len(allocations)):
    #     if neg[i] < multiplier:
    #         v = -1
    #     if np.abs(neg[i] - multiplier) <= 1e-5:
    #         v = np.random.uniform(-1, 1)
    #     if neg[i] > multiplier > pos[i]:
    #         v = 0
    #     if np.abs(pos[i] - multiplier) <= 1e-5:
    #         v = np.random.uniform(-1, 1)
    #     if pos[i] > multiplier:
    #         v = 1
    #     allocations_try[i] = v

    for i in range(len(allocations)):

        # print((neg[i] + pos[i]) / 2)
        thresh = (neg[i] + pos[i]) / 2
        # print(thresh)

        if np.abs(multiplier - thresh) <= 1e-2:
            v = np.random.choice([-1, 1])  # but its actually either -1 or 1...
        elif multiplier > thresh:
            v = -1
        else:
            v = 1
        allocations_try[i] = v

    # for i in range(len(allocations)):

    #     # print((neg[i] + pos[i]) / 2)
    #     thresh = (neg[4000] + pos[4000]) / 2

    #     if pos[i] > thresh:
    #         v = 1
    #     elif neg[i] > thresh:
    #         v = -1
    #     else:
    #         v = 0
    #     allocations_try[i] = v

    # ax.plot(midpoints, allocations_try, color="red")

    transfers = mechanism._allocations_to_transfers(allocations)
    externalities = mechanism._allocations_to_externalities(allocations, tau, prob_state0)

    allocations_noinfo = np.ones(len(allocations))
    allocations_noinfo[:50] = -1
    transfers_noinfo = mechanism._allocations_to_transfers(allocations_noinfo)
    externalities_noinfo = mechanism._allocations_to_externalities(
        allocations_noinfo, tau, prob_state0
    )

    from scipy.integrate import trapezoid

    def calc_mean(x):
        return trapezoid(x * dist.pdf(mechanism.midpoints), x=mechanism.midpoints)

    dx = np.diff(midpoints)[0]
    transfers_mean = calc_mean(transfers)
    externalities_mean = calc_mean(externalities)

    transfers_mean_noinfo = calc_mean(transfers_noinfo)
    externalities_mean_noinfo = calc_mean(externalities_noinfo)

    print(transfers_mean, externalities_mean, transfers_mean - externalities_mean)
    print(
        transfers_mean_noinfo,
        externalities_mean_noinfo,
        transfers_mean_noinfo - externalities_mean_noinfo,
    )

    # # epsilon = 1e-8

    # # # 1. Identify “random” elements by checking if they are not near -1, 0, or 1.
    # # mask = (
    # #     (np.abs(allocations_virtuals_pool + 1) > epsilon)
    # #     & (np.abs(allocations_virtuals_pool - 0) > epsilon)
    # #     & (np.abs(allocations_virtuals_pool - 1) > epsilon)
    # # )

    # # 2. Sum of the known portion of the array:
    # sum_known = np.sum(allocations_virtuals_pool[~mask])
    # # 3. Solve for C such that sum(allocations) = 0
    # #    Let M be the number of “random” elements.  We replace them all by C.
    # #    Then the total sum = sum_known + M*C.  We want this = 0.
    # #    => C = -sum_known / M
    # M = mask.sum()  # number of random elements
    # C = -sum_known / M
    # # 4. Assign that constant to the “random” elements
    # allocations_virtuals_pool[mask] = C
    # # ax.plot(midpoints, allocations_virtuals_pool, label="pooling")

    ax.legend()
    fig.savefig(savedir / "allocations.pdf")

    fig, ax = plt.subplots()
    ax.plot(midpoints, externalities)
    ax.plot(midpoints, externalities_noinfo)
    fig.savefig(savedir / "externalities.pdf")

    fig, ax = plt.subplots()
    ax.plot(midpoints, transfers)
    ax.plot(midpoints, transfers_noinfo)
    fig.savefig(savedir / "transfers.pdf")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    virtual_values = VirtualValues(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    )
    ax.plot(
        midpoints,
        virtual_values.negative,
        color="k",
        ls="solid",
        label=r"$\phi^{-}$",
    )

    ax.plot(
        midpoints,
        virtual_values.positive,
        color="blue",
        ls="solid",
        label=r"$\phi^{+}$",
    )

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

    virtual_values = VirtualValues(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    )

    # i, j = 250, 750
    # ax.fill_between(
    #     midpoints[:i],
    #     virtual_values.negative[:i],
    #     virtual_values.positive[:i],
    #     color="y",
    #     alpha=0.3,
    # )
    # ax.fill_between(
    #     midpoints[i:j],
    #     virtual_values.negative[i:j],
    #     virtual_values.positive[i:j],
    #     color="b",
    #     alpha=0.3,
    # )
    # ax.fill_between(
    #     midpoints[j:],
    #     virtual_values.negative[j:],
    #     virtual_values.positive[j:],
    #     color="y",
    #     alpha=0.3,
    # )

    # ax.fill_between(midpoints[:250], 2, -1, color="y", alpha=0.3)
    # ax.fill_between(midpoints[250:750], 2, -1, color="b", alpha=0.3)
    # ax.fill_between(midpoints[750:], 2, -1, color="y", alpha=0.3)

    # ax.axhline(y=0.05, color="red", label="$\lambda$")
    # ax.axvline(x=i / 1000, color="k", ls="dashed")
    # ax.axvline(x=j / 1000, color="k", ls="dashed")
    ax.legend()
    # ax2 = ax.twinx()
    # ax2.plot(midpoints, allocations, color="red")
    # ax.set_ylim(top=10, bottom=-5)

    ax.set_ylabel("Virtual Value")
    fig.tight_layout()
    fig.savefig(savedir / f"virtuals.pdf", dpi=300)


if __name__ == "__main__":
    main()
