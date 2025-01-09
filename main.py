import numpy as np
from matplotlib import pyplot as plt

from trading_information.distributions import Uniform, BetaMixture
from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOptIncrements, HyperOptVirtuals

if __name__ == "__main__":

    num_intervals = 100
    tau = 1
    prob_state0 = 0
    dist = Uniform(low=0, high=1)
    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots()
    ax.plot(midpoints, dist.pdf(midpoints))
    fig.savefig("./dist.pdf")

    threshold_indices = np.arange(10, 90)  # np.arange(600, 800)
    hyperopt = HyperOptIncrements(mechanism=mechanism, threshold_indices=threshold_indices)

    hyperopt.run(tau, prob_state0)
    threshold_index = hyperopt.best()
    print(threshold_index)

    fig, ax = plt.subplots()
    ax.plot(threshold_indices, hyperopt.objectives)
    fig.savefig("./hyperopt.pdf")

    allocations, transfers, externalities, multiplier, objective = mechanism.solve_with_increments(
        threshold_index, tau, prob_state0
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(midpoints, allocations)

    fig2, ax2 = plt.subplots()
    ax2.plot(midpoints, transfers)

    fig3, ax3 = plt.subplots()
    ax3.plot(midpoints, externalities)

    #
    #
    #
    #
    #

    multipliers = np.linspace(0, 1, 50)
    hyperopt_virtuals = HyperOptVirtuals(mechanism=mechanism, multipliers=multipliers)
    hyperopt_virtuals.run(tau, prob_state0)

    multiplier = hyperopt_virtuals.best()
    print(multiplier)

    allocations, transfers, externalities, objective = mechanism.solve_with_virtuals(
        multiplier, tau, prob_state0, integral_constraint=True
    )

    ax1.plot(midpoints, allocations)
    fig1.savefig("./allocations.pdf")

    ax2.plot(midpoints, transfers)
    fig2.savefig("./transfers.pdf")

    ax3.plot(midpoints, externalities)
    fig3.savefig("./externalities.pdf")
