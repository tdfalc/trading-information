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

    # # def solve(self, tau0: float, tau1: float, prob_state0: float, threshold_type: float):
    def solve(self, alpha: float, threshold_type: float):
        # alpha = tau1 - prob_state0 * (tau0 + tau1)

        tau = 1.5
        tau0 = tau1 = tau
        prob_state0 = 1
        tj = prob_state0

        alpha = tau1 - tj * (tau0 + tau1)

        print("alpha", alpha)

        # alpha = 0

        # Compute virtual values
        # alpha = tau1 - prob_state0 * (tau0 + tau1)
        virtual_values = VirtualValues(self.dist, self.types, alpha=alpha, iron=False)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_types, lb=-1, ub=1)
                binaries = model.addMVar(self.num_types, vtype=gp.GRB.BINARY)
                increments = model.addMVar(len(self.types))
                model.addConstr(increments >= 0)

                # model.addConstr(gp.quicksum(allocations) == 0, name="integral")

                # model.addConstr(gp.quicksum(increments[self.types <= 0.575]) == 1)
                # model.addConstr(gp.quicksum(increments[self.types > 0.575]) == 1)

                objective_value = 0
                for i, type in enumerate(self.types):

                    # model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

                    # model.addConstr(-allocations[i] <= 1 - binaries[i])
                    # model.addConstr(allocations[i] <= binaries[i])

                    # if type < threshold_type:
                    #     model.addConstr(allocations[i] <= 0 - 1e-9)
                    # else:
                    #     model.addConstr(allocations[i] >= 0 + 1e-9)

                    if i > 0:  # Dont think we need this with ironed virtual values
                        # model.addConstr(allocations[i] >= allocations[i - 1])

                        # if type <= threshold_type:
                        #     model.addConstr(allocations[i] == -0.34666667)
                        # else:
                        #     model.addConstr(allocations[i] == 1)

                        virtual_value = (
                            virtual_values.negative[i] * (1 - binaries[i])
                            + virtual_values.positive[i] * binaries[i]
                        )
                        objective_value += virtual_value * allocations[i]

                        # # pooling property
                        # if self.types[i] < break_type:
                        #     model.addConstr(allocation[i] == allocation[i - 1])
                        # if 1 - binaries[i]:
                        #     model.addConstr(allocations[i] == allocations[i - 1])
                        # N = 0.75
                        # if type >= 0.75:
                        #     model.addConstr(allocations[i] == 1)
                        # else:
                        #     model.addConstr(allocations[i] <= 0)

                        extra_ex = tau0 * tj * self._pdfs[i]
                        extra_ex += alpha * self._pdfs[i]
                        extra_ex -= alpha * type * self._pdfs[i]

                        objective_value -= extra_ex

                model.setObjective(objective_value * self.step, GRB.MAXIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)

                print("objective", model.ObjVal)
                print(np.sum(allocations))
                print(np.mean(transfers))

                multiplier = 1  # model.getConstrByName("integral").Pi
                print("status", model.status)

                print("objective", model.ObjVal)
                print("sumalls", np.sum(allocations), np.mean(allocations))
                print("transfers", np.mean(transfers), np.min(transfers), np.max(transfers))
                print(allocations[:10])

                # print(allocations[:350])

                return allocations, transfers, multiplier

    def solve(self, alpha: float, threshold_type: float):
        # alpha = tau1 - prob_state0 * (tau0 + tau1)

        # Compute virtual values
        # alpha = tau1 - prob_state0 * (tau0 + tau1)

        tau = 100
        tau0 = tau1 = tau
        prob_state0 = 1
        tj = prob_state0

        alpha = tau1 - tj * (tau0 + tau1)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                increments = model.addMVar(len(self.types))
                mask = self.types >= 0
                # model.addConstr(
                #     gp.quicksum(increments[mask] * self._count_types) == self._sum_types
                # )
                model.addConstr(increments >= 0)

                # Auxiliary variable to represent the resultant allocation based on the increments.
                allocation = model.addMVar(self.num_types, lb=-1, ub=1)
                binaries = model.addMVar(self.num_types, vtype=gp.GRB.BINARY)

                model.addConstr(gp.quicksum(allocation) == 0, name="integral")
                # model.addConstr(gp.quicksum(increments[self.types <= 0.75]) == 1)
                # model.addConstr(gp.quicksum(increments[self.types > 0.75]) == 1)

                avg_transfer, avg_externality = 0, 0
                for i in range(self.num_types):

                    model.addConstr(allocation[i] == -1 + gp.quicksum(increments[: i + 1]))
                    model.addConstr(-allocation[i] <= 1 - binaries[i])
                    model.addConstr(allocation[i] <= binaries[i])

                    # if self.types[i] <= 0.5:
                    #     model.addConstr(allocation[i] == -1)
                    # else:
                    #     model.addConstr(allocation[i] == 1)

                    if i > 0:
                        externality = 0
                        # alpha = 0
                        # if i >= 0:
                        avg_transfer += binaries[i] * -allocation[i] * self._pdfs[i]
                        avg_transfer += allocation[i] * (
                            self._pdfs[i] * self.types[i] + self._cdfs[i]
                        )

                        externality = 1 - self.types[i] * (1 + allocation[i])
                        externality -= binaries[i] * (1 - 2 * self.types[i]) * allocation[i]
                        externality *= alpha
                        externality += tau0 * prob_state0

                        # if self.types[i] >= 0.75:
                        #     model.addConstr(allocation[i] == 1)

                        avg_externality += externality

                        # N = 0.65
                        # if self.types[i] >= N:
                        #     model.addConstr(allocation[i] == allocation[i - 1])

                objective_value = self.step * (avg_transfer - avg_externality)

                model.setObjective(objective_value, GRB.MAXIMIZE)
                model.optimize()

                allocations = allocation.X

                multiplier = 1  # model.getConstrByName("integral").Pi
                # allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)

                print("objective", model.ObjVal)
                print("sumalls", np.sum(allocations))
                print("transfers", np.mean(transfers), np.min(transfers), np.max(transfers))
                print(allocations[:10])

                return allocations, transfers, multiplier

    def solve(self, alpha: float, threshold_type: float):

        tau = 5

        tau0 = tau1 = tau
        prob_state0 = 1
        tj = prob_state0

        threshold_type = 0.75

        alpha = tau1 - tj * (tau0 + tau1)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                increments = model.addMVar(len(self.types))
                mask = self.types >= 0
                model.addConstr(
                    gp.quicksum(increments[mask] * self._count_types) == self._sum_types
                )
                model.addConstr(increments >= 0)

                # These additional constraints ensure that the allocation function before and after
                # the fully informative type integrate to 1.
                model.addConstr(gp.quicksum(increments[self.types <= threshold_type]) == 1)
                model.addConstr(gp.quicksum(increments[self.types > threshold_type]) == 1)

                avg_transfer, avg_externality = 0, 0
                for i in range(self.num_types):
                    mask = self.types >= np.maximum(self.types[i], threshold_type)

                    avg_transfer += increments[i] * (self.types * self._pdfs + self._cdfs)[i:].sum()
                    avg_transfer -= increments[i] * self._pdfs[mask].sum()

                    avg_externality -= increments[i] * (self.types * self._pdfs)[i:].sum()
                    avg_externality += (
                        increments[i] * ((2 * self.types - 1) * self._pdfs)[mask].sum()
                    )

                    # externality = -increments[i] * (self.types * self._pdfs)[i:].sum()
                    # externality += increments[i] * ((2 * self.types - 1) * self._pdfs)[mask].sum()
                    # externality *= alpha

                    # avg_externality += externality

                # These terms are constant with respect to the optimsation variables, however we
                # add them in we can sanity check the objective function value. Can delete.
                mask = self.types > 0
                avg_transfer -= np.sum(self.types[mask] * self._pdfs[mask] + self._cdfs[mask])
                avg_transfer += np.sum(self._pdfs[self.types >= threshold_type])

                mask = self.types >= np.maximum(self.types, threshold_type)
                avg_externality += np.sum((1 - 2 * self.types[mask]) * self._pdfs[mask])
                avg_externality += np.sum(self._pdfs[self.types > 0])
                avg_externality *= tau1 - prob_state0 * (tau0 + tau1)
                avg_externality += tau0 * prob_state0 * self.num_types

                objective_value = self.step * (avg_transfer - avg_externality)

                model.setObjective(objective_value, GRB.MAXIMIZE)
                model.optimize()

                allocations = self._convert_increments_to_allocations(increments.X)

                multiplier = 1  # model.getConstrByName("integral").Pi
                # allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)

                print("objective", model.ObjVal)
                print("sumalls", np.mean(allocations), np.sum(allocations))
                print("transfers", np.mean(transfers), np.min(transfers), np.max(transfers))

                return allocations, transfers, multiplier
