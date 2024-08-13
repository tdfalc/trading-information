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

    # def solve(self, tau0: float, tau1: float, prob_state0: float, threshold_type: float):
    def solve(self, alpha: float, threshold_type: float):
        # alpha = tau1 - prob_state0 * (tau0 + tau1)

        # Compute virtual values
        # alpha = tau1 - prob_state0 * (tau0 + tau1)
        virtual_values = VirtualValues(self.dist, self.types, alpha=alpha, iron=True)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                # increments = model.addMVar(self.num_types)
                allocations = model.addMVar(self.num_types, lb=-1, ub=1)

                # model.addConstr(
                #     gp.quicksum(increments[self.types >= 0] * self._count_types) == self._sum_types
                # )
                # model.addConstr(increments >= 0)

                model.addConstr(gp.quicksum(allocations) == 0, name="integral")

                objective_value = 0
                for i, type in enumerate(self.types):

                    # model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

                    if self.types[i] < threshold_type:
                        model.addConstr(allocations[i] <= 0)
                    else:
                        model.addConstr(allocations[i] >= 0)

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1])

                    if type <= threshold_type:
                        objective_value += virtual_values.negative[i] * allocations[i]

                    else:
                        objective_value += virtual_values.positive[i] * allocations[i]

                # increments = model.addMVar(len(self.types))
                # model.addConstr(
                #     gp.quicksum(increments[self.types >= 0] * self._count_types) == self._sum_types
                # )
                # model.addConstr(increments >= 0)

                # allocations = model.addMVar(self.num_types, lb=-1, ub=1)
                # virtual_value = model.addMVar(self.num_types, lb=-100, ub=100)

                # objective_value = 0
                # for i in range(self.num_types):

                #     model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

                #     if self.types[i] < threshold_type:
                #         model.addConstr(allocations[i] <= 0)
                #     else:
                #         model.addConstr(allocations[i] >= 0)

                #     type = self.types[i]
                #     phi_neg = type * self._pdfs[i] + self._cdfs[i]
                #     phi_pos = (type - 1) * self._pdfs[i] + self._cdfs[i]

                #     if self.types[i] <= threshold_type:
                #         model.addConstr(virtual_value[i] == phi_neg)
                #     else:
                #         model.addConstr(virtual_value[i] == phi_pos)

                #     if i > 0:
                #         objective_value += virtual_value[i] * allocations[i]

                model.setObjective(objective_value * self.step, GRB.MAXIMIZE)
                model.optimize()

                multiplier = 1  # model.getConstrByName("integral").Pi
                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)

                print(np.sum(allocations) / len(self.types))

                return allocations, transfers, multiplier
