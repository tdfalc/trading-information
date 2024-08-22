from typing import Callable, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from trading_information.typing import _Floats
from trading_information.virtual_values import VirtualValues
from trading_information.distributions import Distribution


class Mechanism:
    def __init__(self, num_types: int, dist: Distribution) -> None:
        self.num_types = num_types
        self.dist = dist

        self.types = np.linspace(0, 1, self.num_types)
        self.step = 1 / (self.num_types - 1)

        # Precompute density function evaluation for each type
        self._pdfs = self.dist.pdf(self.types)
        self._cdfs = self.dist.cdf(self.types)

        # Precompute parameters used in constraint calculations
        self._count_types = np.arange(self.num_types, 0, -1)
        self._sum_types = np.sum(self.types > 0)

    def _allocations_to_transfers(self, allocation: _Floats) -> _Floats:
        return (
            self.types * allocation
            + np.minimum(-allocation, 0)
            - np.cumsum(allocation) / self.num_types
        )

    def _convert_increments_to_allocations(self, increments: _Floats) -> _Floats:
        return np.cumsum(increments) - 1

    def solve(self, tau: float, prob_state0: float, threshold_type: float):

        alpha = tau * (1 - 2 * prob_state0)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_types, lb=-1, ub=1)
                model.addConstr(gp.quicksum(allocations) == 0, name="integral")

                objective_value = 0
                for i, type in enumerate(self.types):

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

                        if type <= threshold_type:
                            model.addConstr(allocations[i] <= 0)

                        if type >= threshold_type:
                            model.addConstr(allocations[i] >= 0)

                        transfer = 0
                        if type >= threshold_type:
                            transfer += -allocations[i] * self._pdfs[i]
                        transfer += allocations[i] * (self._pdfs[i] * self.types[i] + self._cdfs[i])

                        externality = 1 - self.types[i] * (1 + allocations[i])
                        if type >= threshold_type:
                            externality -= (1 - 2 * self.types[i]) * allocations[i]
                        externality *= alpha
                        externality += tau * prob_state0

                        objective_value += transfer - externality

                model.setObjective(objective_value * self.step, GRB.MAXIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                multiplier = model.getConstrByName("integral").Pi
                # print("status", model.status)

                # print("objective", model.ObjVal)
                # print(
                #     "allocation mean", np.mean(allocations), "allocation sum", np.sum(allocations)
                # )
                # print("transfers", np.mean(transfers))

                # print(allocations[:350])

                return allocations, transfers, multiplier, model.ObjVal
