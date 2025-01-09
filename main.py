import numpy as np
from matplotlib import pyplot as plt

from trading_information.distributions import Uniform
from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOptIncrements

if __name__ == "__main__":

    num_intervals = 100
    tau = 0
    prob_state0 = 0.5
    dist = Uniform(low=0, high=1)

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    threshold_indices = np.arange(50, 60)
    hyperopt = HyperOptIncrements(
        mechanism=mechanism, threshold_indices=threshold_indices
    )

    hyperopt.run(tau, prob_state0)
    threshold_index = hyperopt.best()

    fig, ax = plt.subplots()
    ax.plot(threshold_indices, hyperopt.objectives)
    fig.savefig("./hyperopt.pdf")

    allocations, transfers, externalities, multiplier, objective = (
        mechanism.solve_with_increments(tau, prob_state0, threshold_index)
    )

    fig, ax = plt.subplots()
    ax.plot(midpoints, allocations)
    fig.savefig("./allocations.pdf")

    fig, ax = plt.subplots()
    ax.plot(midpoints, transfers)
    fig.savefig("./transfers.pdf")

    fig, ax = plt.subplots()
    ax.plot(midpoints, externalities)
    fig.savefig("./externalities.pdf")
