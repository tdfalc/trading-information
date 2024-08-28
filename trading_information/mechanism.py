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

    def _allocations_to_transfers(self, allocations: _Floats) -> _Floats:
        return (
            self.types * allocations
            + np.minimum(-allocations, 0)
            - np.cumsum(allocations) / self.num_types
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

        return externality_for_type(self.types, allocations)

    def _convert_increments_to_allocations(self, increments: _Floats) -> _Floats:
        return np.cumsum(increments) - 1

    # def solve(self, tau: float, prob_state0: float, threshold_type: float):

    #     alpha = tau * (1 - 2 * prob_state0)

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             allocations = model.addMVar(self.num_types, lb=-1, ub=1)
    #             model.addConstr(gp.quicksum(allocations) == 0, name="integral")

    #             avg_transfer, avg_externality = 0, 0
    #             for i, type in enumerate(self.types):

    #                 if i > 0:
    #                     model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

    #                     if type <= threshold_type:
    #                         model.addConstr(allocations[i] <= 0)

    #                     if type >= threshold_type:
    #                         model.addConstr(allocations[i] >= 0)

    #                     if type >= threshold_type:
    #                         avg_transfer += -allocations[i] * self._pdfs[i]
    #                     avg_transfer += allocations[i] * (
    #                         self._pdfs[i] * self.types[i] + self._cdfs[i]
    #                     )

    #                     externality = 1 - prob_state0 * (1 + allocations[i])
    #                     if type >= threshold_type:
    #                         externality -= (1 - 2 * prob_state0) * allocations[i]
    #                     externality *= alpha
    #                     externality += tau * prob_state0
    #                     avg_externality += externality * self._pdfs[i]

    #             avg_transfer *= self.step
    #             avg_externality *= self.step

    #             model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
    #             model.optimize()

    #             # print("AVG_TRANSFER", avg_transfer.getValue())
    #             # print("AVG_EXTERNALITY", avg_externality.getValue())

    #             allocations = allocations.X
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi

    #             return (
    #                 allocations,
    #                 transfers,
    #                 externalities,
    #                 multiplier,
    #                 avg_transfer.getValue(),
    #                 avg_externality.getValue(),
    #                 model.ObjVal,
    #             )

    #             # return {
    #             #     "allocations": allocations.X,
    #             #     "transfers": self._allocations_to_transfers(allocations.X),
    #             #     "externalities": self._allocations_to_externalities(
    #             #         allocations.X, tau, prob_state0
    #             #     ),
    #             #     "multiplier": model.getConstrByName("integral").Pi,
    #             #     "avg_transfer": avg_transfer.getValue(),
    #             #     "avg_externality": avg_externality.getValue(),
    #             # }

    def solve(self, tau: float, prob_state0: float, threshold_type: float, return_dict=False):

        beta = 0.2

        tau = 0.8

        alpha = tau * (1 - 2 * prob_state0)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_types, lb=-1, ub=1)
                model.addConstr(gp.quicksum(allocations) == 0, name="integral")

                avg_transfer, avg_externality = 0, 0

                u = model.addMVar(self.num_types, lb=0)
                a = model.addVar(lb=-2000, ub=2000)

                cvar_obj = 0
                for i, type in enumerate(self.types):

                    model.addConstr(u[0] == 0)

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

                        if type <= threshold_type:
                            model.addConstr(allocations[i] <= 0)

                        if type >= threshold_type:
                            model.addConstr(allocations[i] >= 0)

                        transfer = type * allocations[i]
                        if type >= threshold_type:
                            transfer -= allocations[i]
                        transfer -= gp.quicksum(allocations[:i] * self.step)
                        avg_transfer += transfer * self._pdfs[i]

                        externality = 1 - prob_state0 * (1 + allocations[i])
                        if type >= threshold_type:
                            externality -= (1 - 2 * prob_state0) * allocations[i]

                        externality *= alpha
                        externality += tau * prob_state0
                        avg_externality += externality * self._pdfs[i]

                        l = externality - transfer

                        model.addConstr(u[i] >= (l - a) * self._pdfs[i])

                cvar_obj = a + self.step / (1 - beta) * gp.quicksum(u)

                avg_externality *= self.step
                avg_transfer *= self.step

                loss = avg_externality - avg_transfer

                lam = 0.9
                model.setObjective(lam * loss + (1 - lam) * cvar_obj, GRB.MINIMIZE)
                model.optimize()

                if return_dict:
                    print("VAR", a.X)
                    print("CVAR", cvar_obj.getValue())
                    print("EV", loss.getValue())

                    return {
                        "allocations": allocations.X,
                        "transfers": self._allocations_to_transfers(allocations.X),
                        "externalities": self._allocations_to_externalities(
                            allocations.X, tau, prob_state0
                        ),
                        "multiplier": model.getConstrByName("integral").Pi,
                        "avg_transfer": avg_transfer.getValue(),
                        "avg_externality": avg_externality.getValue(),
                    }

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
                multiplier = model.getConstrByName("integral").Pi

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                )
