import sys

sys.path.append("../")
sys.path.append("./")

from matplotlib import pyplot as plt
import numpy as np

from trading_information.distributions import BetaMixture, Uniform
from trading_information.virtual_values import VirtualValues
from trading_information.mechanism import Mechanism, MechanismTrapz
from trading_information.hyperopt import HyperOpt

from scipy import stats

dist = Uniform(0, 1)


dist = Uniform(0, 1)

dist = Uniform(0, 1)  # uniform
# dist = stats.expon()
dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))  # bimodal
# dist = BetaMixture([80], [20], [1])  # low
# dist = BetaMixture([20], [80], [1])  # high

mechanism = Mechanism(num_types=1000, dist=dist)
mechanism_trapz = MechanismTrapz(num_intervals=1000, dist=dist)
types = mechanism.types


tau = 0
tau0 = tau1 = tau
prob_state0 = 0
tj = prob_state0
threshold_type = 0.55

(allocations, transfers, externalities, multiplier, avg_transfer, avg_externality, objective) = (
    mechanism.solve(tau, prob_state0, threshold_type)
)

(
    incs,
    allocations_trapz,
    transfers_trapz,
    externalities_trapz,
    multiplier_trapz,
    avg_transfer_trapz,
    avg_externality_trapz,
    objective_trapz,
) = mechanism_trapz.solve(tau, prob_state0, threshold_type)


fig, ax = plt.subplots()
ax.plot(mechanism.types, allocations)
ax.plot(mechanism_trapz.midpoints, allocations_trapz)
fig.savefig("./test.png", dpi=300)
