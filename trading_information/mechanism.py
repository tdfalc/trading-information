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

    # def solve2(self, tau: float, prob_state0: float, threshold_type: float):

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             self._count_types = np.arange(self.num_types, 0, -1)
    #             self._sum_types = np.sum(self.types > 0) + 1  ## why do i need to add 1 here

    #             increments = model.addMVar(len(self.types), lb=0)
    #             model.addConstr(increments >= 0, name="monotonicity")
    #             model.addConstr(
    #                 gp.quicksum(increments[self.types >= 0] * self._count_types) == self._sum_types,
    #                 name="integral",
    #             )

    #             # model.addConstr(
    #             #     gp.quicksum(increments[1:-1] * (999 - self._count_types[1:-1]))
    #             #     == self.num_types - 2,
    #             #     name="integral",
    #             # )

    #             # These additional constraints ensure that the allocation function before and after
    #             # the threshold type integrate to 1.
    #             # model.addConstr(gp.quicksum(increments[self.types <= threshold_type]) == 1)
    #             # model.addConstr(gp.quicksum(increments[self.types >= threshold_type]) == 1)

    #             # allocation = model.addMVar(self.num_types, lb=-1, ub=1)
    #             # model.addConstr(gp.quicksum(allocation[self.types <= threshold_type]) == -500)
    #             # model.addConstr(gp.quicksum(allocation[self.types >= threshold_type]) == 500)

    #             # allocation = model.addMVar(self.num_types, lb=-1, ub=1)
    #             # model.addConstr(gp.quicksum(allocation[:500]) == -500)
    #             # model.addConstr(gp.quicksum(allocation[500:]) == 500)

    #             avg_transfer, avg_externality = 0, 0
    #             for i in range(self.num_types):
    #                 # model.addConstr(-1 + gp.quicksum(increments[: i + 1]) >= allocation[i])
    #                 # if self.types[i] <= threshold_type:
    #                 #     model.addConstr(-1 + gp.quicksum(increments[: i + 1]) <= 0)
    #                 # else:
    #                 #     model.addConstr(-1 + gp.quicksum(increments[: i + 1]) >= 0)
    #                 model.addConstr(gp.quicksum(increments[: i + 1]) <= 2)

    #                 mask = self.types >= np.maximum(self.types[i], threshold_type)

    #                 avg_transfer += increments[i] * (self.types * self._pdfs + self._cdfs)[i:].sum()
    #                 avg_transfer -= increments[i] * self._pdfs[mask].sum()

    #                 avg_externality -= increments[i] * (prob_state0 * self._pdfs[i:]).sum()
    #                 avg_externality += (
    #                     increments[i] * ((2 * prob_state0 - 1) * self._pdfs[mask]).sum()
    #                 )

    #             # These terms are constant with respect to the optimsation variables, however we
    #             # add them in we can sanity check the objective function value. Can delete.
    #             mask = self.types > 0
    #             avg_transfer -= np.sum(self.types[mask] * self._pdfs[mask] + self._cdfs[mask])
    #             avg_transfer += np.sum(self._pdfs[self.types >= threshold_type])

    #             mask = self.types >= np.maximum(self.types, threshold_type)
    #             avg_externality += np.sum((1 - 2 * prob_state0) * self._pdfs[mask])
    #             avg_externality += np.sum(self._pdfs[self.types > 0])
    #             avg_externality *= tau * (1 - 2 * prob_state0)
    #             avg_externality += tau * prob_state0 * self.num_types

    #             avg_transfer *= self.step
    #             avg_externality *= self.step

    #             model.setObjective(avg_externality - avg_transfer, GRB.MINIMIZE)
    #             model.optimize()

    #             allocations = self._convert_increments_to_allocations(increments.X)
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi

    #             print("SUM", np.sum(allocations[self.types <= threshold_type]))
    #             print("SUM2", np.sum(allocations[self.types >= threshold_type]))
    #             x = increments.X[self.types >= 0] * self._count_types
    #             print("SUM3", x[490:510])
    #             print(self._sum_types)

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

    def solve(self, tau: float, prob_state0: float, threshold_type: float):
        """This is the oriongal one that works"""

        alpha = tau * (1 - 2 * prob_state0)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_types, lb=-1, ub=1)
                model.addConstr(gp.quicksum(allocations) == 0, name="integral")

                avg_transfer, avg_externality = 0, 0
                for i, type in enumerate(self.types):

                    if i > 0:
                        model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

                        if type <= threshold_type:
                            model.addConstr(allocations[i] <= 0)

                        if type >= threshold_type:
                            model.addConstr(allocations[i] >= 0)

                        if type >= threshold_type:
                            avg_transfer += -allocations[i] * self._pdfs[i]
                        avg_transfer += allocations[i] * (
                            self._pdfs[i] * self.types[i] + self._cdfs[i]
                        )

                        externality = 1 - prob_state0 * (1 + allocations[i])
                        if type >= threshold_type:
                            externality -= (1 - 2 * prob_state0) * allocations[i]
                        externality *= alpha
                        externality += tau * prob_state0
                        avg_externality += externality * self._pdfs[i]

                avg_transfer *= self.step
                avg_externality *= self.step

                model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
                model.optimize()

                # print("AVG_TRANSFER", avg_transfer.getValue())
                # print("AVG_EXTERNALITY", avg_externality.getValue())

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

                # return {
                #     "allocations": allocations.X,
                #     "transfers": self._allocations_to_transfers(allocations.X),
                #     "externalities": self._allocations_to_externalities(
                #         allocations.X, tau, prob_state0
                #     ),
                #     "multiplier": model.getConstrByName("integral").Pi,
                #     "avg_transfer": avg_transfer.getValue(),
                #     "avg_externality": avg_externality.getValue(),
                # }

    # def solve(self, tau: float, prob_state0: float, threshold_type: float, return_dict=False):

    #     beta = 0.99

    #     # tau = 0.8

    #     alpha = tau * (1 - 2 * prob_state0)

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             allocations = model.addMVar(self.num_types, lb=-1, ub=1)
    #             model.addConstr(gp.quicksum(allocations) == 0, name="integral")

    #             # model.addConstr(allocations[250:] == 1 / 3)
    #             # model.addConstr(allocations[:250] == -1)

    #             avg_transfer, avg_externality = 0, 0

    #             u = model.addMVar(self.num_types, lb=0)
    #             a = model.addVar(lb=-2000, ub=2000)

    #             cvar_obj = 0
    #             for i, type in enumerate(self.types):

    #                 model.addConstr(u[0] == 0)

    #                 if i > 0:
    #                     model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

    #                     if type <= threshold_type:
    #                         model.addConstr(allocations[i] <= 0)

    #                     if type >= threshold_type:
    #                         model.addConstr(allocations[i] >= 0)

    #                     transfer = type * allocations[i]
    #                     if type >= threshold_type:
    #                         transfer -= allocations[i]
    #                     transfer -= gp.quicksum(allocations[:i] * self.step)
    #                     avg_transfer += transfer * self._pdfs[i]

    #                     externality = 1 - prob_state0 * (1 + allocations[i])
    #                     if type >= threshold_type:
    #                         externality -= (1 - 2 * prob_state0) * allocations[i]

    #                     externality *= alpha
    #                     externality += tau * prob_state0
    #                     avg_externality += externality * self._pdfs[i]

    #                     l = externality - transfer

    #                     model.addConstr(u[i] >= (l - a) * self._pdfs[i])

    #             cvar_obj = a + self.step / (1 - beta) * gp.quicksum(u)

    #             avg_externality *= self.step
    #             avg_transfer *= self.step

    #             loss = avg_externality - avg_transfer

    #             lam = 1
    #             model.setObjective(lam * loss + (1 - lam) * cvar_obj, GRB.MINIMIZE)
    #             model.optimize()

    #             if return_dict:
    #                 print("VAR", a.X)
    #                 print("CVAR", cvar_obj.getValue())
    #                 print("EV", loss.getValue())

    #                 return {
    #                     "allocations": allocations.X,
    #                     "transfers": self._allocations_to_transfers(allocations.X),
    #                     "externalities": self._allocations_to_externalities(
    #                         allocations.X, tau, prob_state0
    #                     ),
    #                     "multiplier": model.getConstrByName("integral").Pi,
    #                     "avg_transfer": avg_transfer.getValue(),
    #                     "avg_externality": avg_externality.getValue(),
    #                 }

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


class MechanismTrapz:
    def __init__(self, num_intervals: int, dist: Distribution) -> None:

        self.num_intervals = num_intervals
        self.dist = dist

        self.types = np.linspace(0, 1, self.num_intervals + 1)
        self.step = 1 / self.num_intervals
        self.midpoints = (self.types[:-1] + self.types[1:]) / 2

        # Precompute density function evaluation for each midpoint type
        self._pdfs = self.dist.pdf(self.midpoints)
        self._cdfs = self.dist.cdf(self.midpoints)

        # # Precompute parameters used in constraint calculations
        # self._count_types = np.arange(self.num_types, 0, -1)
        # self._sum_types = np.sum(self.types > 0)

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

    def solve(self, tau: float, prob_state0: float, threshold_type: float):

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                increments = model.addMVar(self.num_intervals, lb=0)
                model.addConstr(increments >= 0, name="monotonicity")
                model.addConstr(
                    gp.quicksum(increments * np.arange(self.num_intervals, 0, -1))
                    == self.num_intervals,
                    name="integral",
                )

                ## How best to implement this threshold...

                # model.addConstr(gp.quicksum(increments) == 2)

                # These additional constraints ensure that the allocation function before and after
                # the threshold type integrate to 1.
                # model.addConstr(gp.quicksum(increments[self.midpoints <= threshold_type]) == 1)
                # model.addConstr(gp.quicksum(increments[self.midpoints >= threshold_type]) == 1)

                slack = model.addVar(lb=0, ub=1)
                model.addConstr(
                    gp.quicksum(increments[self.midpoints < threshold_type]) == 1 - slack
                )
                model.addConstr(
                    gp.quicksum(increments[self.midpoints >= threshold_type]) == 1 + slack
                )

                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    # if midpoint <= threshold_type:
                    #     model.addConstr(-1 + gp.quicksum(increments[: i + 1]) <= 0)
                    # else:
                    #     model.addConstr(-1 + gp.quicksum(increments[: i + 1]) >= 0)

                    mask = self.midpoints >= np.maximum(midpoint, threshold_type)

                    # Update expression for average transfer
                    transfer = increments[i] * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
                    transfer -= increments[i] * self._pdfs[mask].sum()
                    avg_transfer += transfer

                    # Update expression for average externality
                    externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
                    externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
                    avg_externality += externality

                # These terms are constant with respect to the optimsation variables, however we
                # add them in we can sanity check the objective function value. Can delete.
                avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                avg_transfer += self._pdfs[self.midpoints >= threshold_type].sum()
                avg_transfer *= self.step

                avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                    self.midpoints >= threshold_type
                ].sum()
                avg_externality += self._pdfs.sum()
                avg_externality *= tau * (1 - 2 * prob_state0)
                avg_externality += tau * prob_state0 * self._pdfs.sum()
                avg_externality *= self.step

                model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
                model.optimize()

                allocations = self._convert_increments_to_allocations(increments.X)
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
                multiplier = model.getConstrByName("integral").Pi

                print("SLACK", slack.X)

                return (
                    increments.X,
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                )

                # return {
                #     "allocations": allocations.X,
                #     "transfers": self._allocations_to_transfers(allocations.X),
                #     "externalities": self._allocations_to_externalities(
                #         allocations.X, tau, prob_state0
                #     ),
                #     "multiplier": model.getConstrByName("integral").Pi,
                #     "avg_transfer": avg_transfer.getValue(),
                #     "avg_externality": avg_externality.getValue(),
                # }

    def solve(self, tau: float, prob_state0: float, threshold_type: float):

        threshold_index = 550

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                increments = model.addMVar(self.num_intervals, lb=0)
                model.addConstr(increments >= 0, name="monotonicity")
                model.addConstr(
                    gp.quicksum(increments * np.arange(self.num_intervals, 0, -1))
                    == self.num_intervals,
                    name="integral",
                )

                ## How best to implement this threshold...
                slack = model.addVar(lb=0, ub=1)

                # model.addConstr(gp.quicksum(increments) == 2)

                # These additional constraints ensure that the allocation function before and after
                # the threshold type integrate to 1.
                model.addConstr(gp.quicksum(increments[:threshold_index]) <= 1)
                model.addConstr(gp.quicksum(increments[: threshold_index + 1]) >= 1)
                model.addConstr(gp.quicksum(increments) <= 2)

                # model.addConstr(gp.quicksum(increments[threshold_index:]) == 1 + slack)

                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    # threshold_type = self.midpoints[threshold_index]

                    mask = self.midpoints >= np.maximum(midpoint, threshold_type)

                    # Update expression for average transfer
                    transfer = increments[i] * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
                    transfer -= increments[i] * self._pdfs[mask].sum()
                    avg_transfer += transfer

                    # Update expression for average externality
                    externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
                    externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
                    avg_externality += externality

                # These terms are constant with respect to the optimsation variables, however we
                # add them in we can sanity check the objective function value. Can delete.
                avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                avg_transfer += self._pdfs[self.midpoints >= threshold_type].sum()
                avg_transfer *= self.step

                avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                    self.midpoints >= threshold_type
                ].sum()
                avg_externality += self._pdfs.sum()
                avg_externality *= tau * (1 - 2 * prob_state0)
                avg_externality += tau * prob_state0 * self._pdfs.sum()
                avg_externality *= self.step

                model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
                model.optimize()

                allocations = self._convert_increments_to_allocations(increments.X)
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
                multiplier = model.getConstrByName("integral").Pi

                return (
                    increments.X,
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                )

                # return {
                #     "allocations": allocations.X,
                #     "transfers": self._allocations_to_transfers(allocations.X),
                #     "externalities": self._allocations_to_externalities(
                #         allocations.X, tau, prob_state0
                #     ),
                #     "multiplier": model.getConstrByName("integral").Pi,
                #     "avg_transfer": avg_transfer.getValue(),
                #     "avg_externality": avg_externality.getValue(),
                # }

    # def solve(self, tau: float, prob_state0: float, threshold_type: float):
    #     """This is the oriongal one that works"""

    #     alpha = tau * (1 - 2 * prob_state0)

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
    #             model.addConstr(gp.quicksum(allocations) == 0, name="integral")

    #             avg_transfer, avg_externality = 0, 0
    #             for i, midpoint in enumerate(self.midpoints):

    #                 if i > 0:
    #                     model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

    #                 if midpoint <= threshold_type:
    #                     model.addConstr(allocations[i] <= 0)

    #                 if midpoint >= threshold_type:
    #                     model.addConstr(allocations[i] >= 0)

    #                 if midpoint >= threshold_type:
    #                     avg_transfer += -allocations[i] * self._pdfs[i]
    #                 avg_transfer += allocations[i] * (
    #                     self._pdfs[i] * self.midpoints[i] + self._cdfs[i]
    #                 )

    #                 externality = 1 - prob_state0 * (1 + allocations[i])
    #                 if midpoint >= threshold_type:
    #                     externality -= (1 - 2 * prob_state0) * allocations[i]
    #                 externality *= alpha
    #                 externality += tau * prob_state0
    #                 avg_externality += externality * self._pdfs[i]

    #             avg_transfer *= self.step
    #             avg_externality *= self.step

    #             model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
    #             model.optimize()

    #             allocations = allocations.X
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi

    #             return (
    #                 1,
    #                 allocations,
    #                 transfers,
    #                 externalities,
    #                 multiplier,
    #                 avg_transfer.getValue(),
    #                 avg_externality.getValue(),
    #                 model.ObjVal,
    #             )
