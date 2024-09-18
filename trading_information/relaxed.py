from typing import Callable, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from trading_information.typing import _Floats
from trading_information.distributions import Distribution


class Mechanism:
    def __init__(self, num_intervals: int, dist: Distribution, lam, beta) -> None:

        self.lam = lam
        self.beta = beta

        self.num_intervals = num_intervals
        self.dist = dist

        self.types = np.linspace(0, 1, self.num_intervals + 1)
        self.step = 1 / self.num_intervals
        self.midpoints = (self.types[:-1] + self.types[1:]) / 2

        # Precompute density function evaluation for each midpoint type
        self._pdfs = self.dist.pdf(self.midpoints)
        self._cdfs = self.dist.cdf(self.midpoints)

    def _allocations_to_transfers(self, allocations: _Floats) -> _Floats:
        return (
            self.midpoints * allocations
            + np.minimum(-allocations, 0)
            - np.cumsum(allocations) / self.num_intervals
        )

    def _allocations_to_externalities(
        self, allocations: _Floats, tau: float, prob_state0
    ) -> _Floats:

        # def externality_for_type(type, x):
        #     ps1 = 1 - type - type * x + (1 - 2 * type) * np.minimum(0, -x)
        #     externality = ps1
        #     externality *= 1 - 2 * prob_state0
        #     # externality *= prob_state0
        #     externality *= tau

        #     return externality + tau * prob_state0

        def externality_for_type(type, x):
            ps1 = 1 - prob_state0 - prob_state0 * x + (1 - 2 * prob_state0) * np.minimum(0, -x)
            externality = ps1
            externality *= 1 - 2 * prob_state0
            # externality *= prob_state0
            externality *= tau

            return externality + tau * prob_state0

        return externality_for_type(self.midpoints, allocations)

    def _convert_increments_to_allocations(self, increments: _Floats) -> _Floats:
        return np.cumsum(increments) - 1

    def solve_high(self, tau: float, prob_state0: float, threshold_index: int, do_print=False):

        beta = self.beta

        self.types = np.linspace(0.5, 1, self.num_intervals + 1)
        self.step = 1 / self.num_intervals
        self.midpoints = (self.types[:-1] + self.types[1:]) / 2

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
                # model.addConstr(gp.quicksum(allocations) == 1)
                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    threshold = self.midpoints[threshold_index]

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1])

                    transfer2 = midpoint * allocations[i]
                    if midpoint >= threshold:
                        transfer2 -= allocations[i]
                    transfer2 -= gp.quicksum(allocations[:i] * self.step)

                    # Compute actual externality
                    externality2 = 1 - prob_state0 * (1 + allocations[i])
                    if midpoint >= threshold:
                        externality2 -= (1 - 2 * prob_state0) * allocations[i]
                    externality2 *= tau * (1 - 2 * prob_state0)
                    externality2 += tau * prob_state0

                    avg_transfer += transfer2 * self.step * self._pdfs[i]
                    avg_externality += externality2 * self.step * self._pdfs[i]

                model.setObjective(avg_externality - avg_transfer, GRB.MINIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)

                return (allocations, transfers, externalities)

    def solve_low(self, tau: float, prob_state0: float, threshold_index: int, do_print=False):

        beta = self.beta

        self.types = np.linspace(0, 0.5, self.num_intervals + 1)
        self.step = 1 / self.num_intervals
        self.midpoints = (self.types[:-1] + self.types[1:]) / 2

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
                # model.addConstr(gp.quicksum(allocations) == -1)
                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    threshold = self.midpoints[threshold_index]

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1])

                    transfer2 = midpoint * allocations[i]
                    if midpoint >= threshold:
                        transfer2 -= allocations[i]
                    transfer2 -= gp.quicksum(allocations[:i] * self.step)

                    # Compute actual externality
                    externality2 = 1 - prob_state0 * (1 + allocations[i])
                    if midpoint >= threshold:
                        externality2 -= (1 - 2 * prob_state0) * allocations[i]
                    externality2 *= tau * (1 - 2 * prob_state0)
                    externality2 += tau * prob_state0

                    avg_transfer += transfer2 * self.step * self._pdfs[i]
                    avg_externality += externality2 * self.step * self._pdfs[i]

                model.setObjective(avg_externality - avg_transfer, GRB.MINIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)

                return (allocations, transfers, externalities)
