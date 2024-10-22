from typing import Callable, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from trading_information.typing import _Floats
from trading_information.distributions import Distribution
from trading_information.virtual_values_basic import VirtualValues


class Mechanism:
    def __init__(self, num_intervals: int, dist: Distribution) -> None:

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

    def solve(self, tau: float, prob_state0: float, threshold_index: int):

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

                # These additional constraints ensure that the allocation function
                # is negative before the threshold index, and nonnegative afterwards.
                model.addConstr(gp.quicksum(increments[:threshold_index]) <= 1)
                model.addConstr(gp.quicksum(increments[: threshold_index + 1]) >= 1)
                model.addConstr(gp.quicksum(increments) <= 2)

                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    threshold = self.midpoints[threshold_index]
                    mask = self.midpoints >= np.maximum(midpoint, threshold)

                    # Update expression for average transfer
                    transfer = increments[i] * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
                    transfer -= increments[i] * self._pdfs[mask].sum()
                    avg_transfer += transfer  # * self._pdfs[i]

                    # Update expression for average externality
                    externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
                    externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
                    avg_externality += externality  # * self._pdfs[i]

                # These terms are constant with respect to the optimsation variables, however we
                # add them in we can sanity check the objective function value. Can delete.
                avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
                avg_transfer *= self.step

                avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                    self.midpoints >= threshold
                ].sum()
                avg_externality += self._pdfs.sum()
                avg_externality *= tau * (1 - 2 * prob_state0)
                avg_externality += tau * prob_state0 * self._pdfs.sum()
                avg_externality *= self.step

                model.setObjective((avg_transfer - avg_externality), GRB.MAXIMIZE)
                model.optimize()

                allocations = self._convert_increments_to_allocations(increments.X)
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
                multiplier = model.getConstrByName("integral").Pi * self.num_intervals

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                )

    # def solve_noinfo(self, tau: float, prob_state0: float, threshold_index: int):

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             increments = model.addMVar(self.num_intervals, lb=0)
    #             model.addConstr(increments >= 0, name="monotonicity")
    #             model.addConstr(
    #                 gp.quicksum(increments * np.arange(self.num_intervals, 0, -1))
    #                 == self.num_intervals,
    #                 name="integral",
    #             )

    #             # These additional constraints ensure that the allocation function
    #             # is negative before the threshold index, and nonnegative afterwards.
    #             # model.addConstr(gp.quicksum(increments[:threshold_index]) <= 1)
    #             # model.addConstr(gp.quicksum(increments[: threshold_index + 1]) >= 1)
    #             model.addConstr(gp.quicksum(increments) <= 2)

    #             allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
    #             # model.addConstr(allocations[:50] == -1)
    #             # model.addConstr(allocations[50:] == 1)

    #             model.addConstr(allocations[:50] == -1)
    #             model.addConstr(allocations[50:] == 1)

    #             avg_transfer, avg_externality = 0, 0
    #             for i, midpoint in enumerate(self.midpoints):

    #                 model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

    #                 threshold = self.midpoints[threshold_index]
    #                 mask = self.midpoints >= np.maximum(midpoint, threshold)

    #                 # Update expression for average transfer
    #                 transfer = increments[i] * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
    #                 transfer -= increments[i] * self._pdfs[mask].sum()
    #                 avg_transfer += transfer  # * self._pdfs[i]

    #                 # Update expression for average externality
    #                 externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
    #                 externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
    #                 avg_externality += externality  # * self._pdfs[i]

    #             # These terms are constant with respect to the optimsation variables, however we
    #             # add them in we can sanity check the objective function value. Can delete.
    #             avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
    #             avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
    #             avg_transfer *= self.step

    #             avg_externality += (1 - 2 * prob_state0) * self._pdfs[
    #                 self.midpoints >= threshold
    #             ].sum()
    #             avg_externality += self._pdfs.sum()
    #             avg_externality *= tau * (1 - 2 * prob_state0)
    #             avg_externality += tau * prob_state0 * self._pdfs.sum()
    #             avg_externality *= self.step

    #             model.setObjective(-(avg_transfer - avg_externality), GRB.MINIMIZE)
    #             model.optimize()

    #             allocations = self._convert_increments_to_allocations(increments.X)
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi * self.num_intervals

    #             return (
    #                 allocations,
    #                 transfers,
    #                 externalities,
    #                 multiplier,
    #                 avg_transfer.getValue(),
    #                 avg_externality.getValue(),
    #                 model.ObjVal,
    #             )

    # def solve(self, tau: float, prob_state0: float, threshold_index: int, do_print=False):

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             increments = model.addMVar(self.num_intervals, lb=0)
    #             model.addConstr(increments >= 0, name="monotonicity")
    #             model.addConstr(
    #                 gp.quicksum(increments * np.arange(self.num_intervals, 0, -1))
    #                 == self.num_intervals,
    #                 name="integral",
    #             )

    #             # These additional constraints ensure that the allocation function
    #             # is negative before the threshold index, and nonnegative afterwards.
    #             # model.addConstr(gp.quicksum(increments[:threshold_index]) == 1)
    #             # model.addConstr(gp.quicksum(increments[: threshold_index + 1]) == 1)
    #             # model.addConstr(gp.quicksum(increments) <= 2)

    #             allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)

    #             # model.addConstr(allocations[:500] == -1)
    #             # model.addConstr(allocations[500:] == 1)

    #             avg_transfer, avg_externality = 0, 0
    #             for i, midpoint in enumerate(self.midpoints):

    #                 threshold = self.midpoints[threshold_index]
    #                 mask = self.midpoints >= np.maximum(midpoint, threshold)

    #                 if i < threshold_index:
    #                     model.addConstr(allocations[i] <= 0)
    #                 else:
    #                     model.addConstr(allocations[i] >= 0)

    #                 # # Update expression for average transfer
    #                 # transfer = increments[i] * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
    #                 # transfer -= increments[i] * self._pdfs[mask].sum()
    #                 # avg_transfer += transfer

    #                 # # Update expression for average externality
    #                 # externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
    #                 # externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
    #                 # avg_externality += externality

    #                 # Compute actual transfer
    #                 model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

    #                 transfer2 = midpoint * allocations[i]
    #                 if midpoint >= threshold:
    #                     transfer2 -= allocations[i]
    #                 transfer2 -= gp.quicksum(allocations[:i] * self.step)

    #                 # Compute actual externality
    #                 externality2 = 1 - prob_state0 * (1 + allocations[i])
    #                 if midpoint >= threshold:
    #                     externality2 -= (1 - 2 * prob_state0) * allocations[i]
    #                 externality2 *= tau * (1 - 2 * prob_state0)
    #                 externality2 += tau * prob_state0

    #                 avg_transfer += transfer2 * self.step * self._pdfs[i]
    #                 avg_externality += externality2 * self.step * self._pdfs[i]

    #             # # These terms are constant with respect to the optimsation variables, however we
    #             # # add them in we can sanity check the objective function value. Can delete.
    #             # avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
    #             # avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
    #             # avg_transfer *= self.step

    #             # avg_externality += (1 - 2 * prob_state0) * self._pdfs[
    #             #     self.midpoints >= threshold
    #             # ].sum()
    #             # avg_externality += self._pdfs.sum()
    #             # avg_externality *= tau * (1 - 2 * prob_state0)
    #             # avg_externality += tau * prob_state0 * self._pdfs.sum()
    #             # avg_externality *= self.step

    #             model.setObjective(avg_externality - avg_transfer, GRB.MINIMIZE)
    #             model.optimize()

    #             if do_print:

    #                 print("TRANSFER", avg_transfer.getValue())
    #                 print("EXTERNALITY", avg_externality.getValue())
    #                 print("model.ObjVal", model.ObjVal)

    #             allocations = self._convert_increments_to_allocations(increments.X)
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi * self.num_intervals

    #             return (
    #                 allocations,
    #                 transfers,
    #                 externalities,
    #                 multiplier,
    #                 avg_transfer.getValue(),
    #                 avg_externality.getValue(),
    #                 model.ObjVal,
    #             )

    # def solve(self, tau: float, prob_state0: float, threshold_index: int, do_print=False):
    #     """
    #     With this formualtion, when the exteranlity outweights the transfer, or we cross the virtual values,
    #     the problem becomes unbounded.
    #     """
    #     beta = 0.999
    #     self.lam = 1

    #     with gp.Env(empty=True) as env:
    #         env.setParam("OutputFlag", 0)
    #         env.start()
    #         with gp.Model(env=env) as model:

    #             allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
    #             model.addConstr(gp.quicksum(allocations) == 0, name="integral")

    #             u = model.addMVar(self.num_intervals, lb=0)
    #             var = model.addVar(lb=-10, ub=10)

    #             y = model.addMVar(self.num_intervals, lb=0)  # allocations positive

    #             avg_transfer, avg_externality = 0, 0
    #             avg_loss = 0
    #             for i, midpoint in enumerate(self.midpoints):
    #                 if i > 0:
    #                     model.addConstr(allocations[i] >= allocations[i - 1], name="monotonicity")

    #                 model.addConstr(y[i] >= allocations[i])

    #                 transfer2 = midpoint * allocations[i]
    #                 transfer2 -= y[i]
    #                 transfer2 -= gp.quicksum(allocations[:i] * self.step)

    #                 # Compute actual externality
    #                 externality2 = 1 - prob_state0 * (1 + allocations[i])
    #                 externality2 -= (1 - 2 * prob_state0) * y[i]
    #                 externality2 *= tau * (1 - 2 * prob_state0)
    #                 externality2 += tau * prob_state0

    #                 # Compute u term in cvar
    #                 r = transfer2 - externality2
    #                 l = -r
    #                 model.addConstr(u[i] >= (l - var) * self._pdfs[i])

    #                 avg_loss += l * self.step * self._pdfs[i]
    #                 avg_transfer += transfer2 * self.step * self._pdfs[i]
    #                 avg_externality += externality2 * self.step * self._pdfs[i]

    #             cvar_obj = var + self.step / (1 - beta) * gp.quicksum(u)

    #             model.setObjective(self.lam * avg_loss + (1 - self.lam) * cvar_obj, GRB.MINIMIZE)
    #             model.optimize()
    #             if do_print:
    #                 print("VAR", var.X)
    #                 print("CVAR", cvar_obj.getValue())
    #                 print("EV", avg_loss.getValue())
    #                 print("TRANSFER", avg_transfer.getValue())
    #                 print("EXTERNALITY", avg_externality.getValue())

    #             allocations = allocations.X
    #             transfers = self._allocations_to_transfers(allocations)
    #             externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
    #             multiplier = model.getConstrByName("integral").Pi * self.num_intervals

    #             # return (
    #             #     allocations,
    #             #     transfers,
    #             #     externalities,
    #             #     multiplier,
    #             #     avg_transfer.getValue(),
    #             #     avg_externality.getValue(),
    #             #     var.X,
    #             #     cvar_obj.getValue(),
    #             #     y.X,
    #             #     model.ObjVal,
    #             # )
    #             return (
    #                 allocations,
    #                 transfers,
    #                 externalities,
    #                 multiplier,
    #                 avg_transfer.getValue(),
    #                 avg_externality.getValue(),
    #                 model.ObjVal,
    #             )

    def solve_virtuals(
        self,
        tau: float,
        prob_state0: float,
        threshold_index: int,
        multi,
        virtual_values,
        do_print=False,
        prescribed_solution=None,
    ):
        """
        With this formualtion, when the exteranlity outweights the transfer, or we cross the virtual values,
        the problem becomes unbounded.
        """

        alpha = tau * (1 - 2 * prob_state0)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
                binaries = model.addMVar(self.num_intervals, vtype=gp.GRB.BINARY)
                increments = model.addMVar(len(self.midpoints))
                # model.addConstr(increments >= 0)

                binaries2 = model.addMVar(self.num_intervals, vtype=gp.GRB.BINARY)

                all_virtuals = model.addMVar(self.num_intervals, lb=-np.inf, ub=np.inf)

                model.addConstr(gp.quicksum(allocations) == 0, name="integral2")

                # model.addConstr(gp.quicksum(increments[self.types <= 0.575]) == 1)
                # model.addConstr(gp.quicksum(increments[self.types > 0.575]) == 1)

                # model.addConstr(allocations[:50] == -1)
                # model.addConstr(allocations[50:] == 1)

                objective_value = 0
                objective_wo_mult = 0
                avg_transfer = 0
                avg_externality = 0
                # first step is to get tehse objectives euql!!!!!!!!

                for i, type in enumerate(self.midpoints):

                    if prescribed_solution is not None:
                        model.addConstr(allocations[i] <= prescribed_solution[i])

                    model.addConstr(allocations[i] == -1 + gp.quicksum(increments[: i + 1]))

                    model.addConstr(-allocations[i] <= 1 - binaries[i])
                    model.addConstr(allocations[i] <= binaries[i])

                    virtual_value = (
                        virtual_values.negative[i] * (1 - binaries[i])
                        + virtual_values.positive[i] * binaries[i]
                    )

                    model.addConstr(all_virtuals[i] == virtual_value)

                    ex = self._pdfs[i] * (2 * tau * (prob_state0**2) - 2 * tau * prob_state0 + tau)

                    LAMBDA = multi
                    objective_value += (virtual_value - LAMBDA) * allocations[i] - ex

                    objective_wo_mult += (virtual_value) * allocations[i] - ex

                    last_virtual = virtual_value

                    # Compute actual transfer
                    midpoint = self.midpoints[i]
                    transfer2 = midpoint * allocations[i]
                    transfer2 -= allocations[i] * binaries[i]
                    transfer2 -= gp.quicksum(allocations[:i] * self.step)

                    # Compute actual externality
                    externality2 = 1 - prob_state0 * (1 + allocations[i])
                    externality2 -= (1 - 2 * prob_state0) * allocations[i] * binaries[i]
                    externality2 *= tau * (1 - 2 * prob_state0)
                    externality2 += tau * prob_state0

                    avg_transfer += transfer2 * self.step * self._pdfs[i]
                    avg_externality += externality2 * self.step * self._pdfs[i]

                    M = 1000
                    if i > 0:
                        q = 1e-4
                        model.addConstr(
                            all_virtuals[i] - all_virtuals[i - 1] <= q + M * (1 - binaries2[i])
                        )
                        model.addConstr(
                            all_virtuals[i] - all_virtuals[i - 1] >= q - M * binaries2[i]
                        )
                        model.addGenConstrIndicator(
                            binaries2[i], 1, allocations[i] - allocations[i - 1] == 0
                        )

                model.setObjective(objective_value * self.step, GRB.MAXIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
                multiplier = LAMBDA  # model.getConstrByName("integral2").Pi * self.num_intervals

                # print("objective_wo_mult", -objective_wo_mult.getValue() * self.step)

                # print(allocations[:350])

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                    binaries.X,
                    all_virtuals.X,
                    binaries2.X,
                )

    def solve_virtuals2(
        self,
        tau: float,
        prob_state0: float,
        threshold_index: int,
        multi,
        virtual_values,
        do_print=False,
        prescribed_solution=None,
        **kwargs,
    ):
        """
        With this formualtion, when the exteranlity outweights the transfer, or we cross the virtual values,
        the problem becomes unbounded.
        """

        allocations = np.zeros(self.num_intervals)
        objective_value = 0
        objective_wo_mult = 0
        avg_transfer = 0
        avg_externality = 0
        for i, type in enumerate(self.midpoints):

            LAMBDA = multi

            if prescribed_solution is not None:
                allocations[i] = prescribed_solution[i]
                if allocations[i] <= 0:
                    obj = virtual_values.negative[i] * allocations[i]  # - LAMBDA * allocations[i]
                else:
                    obj = virtual_values.positive[i] * allocations[i]  # - LAMBDA * allocations[i]
                objective_value += obj * self.step
                ex = self._pdfs[i] * (2 * tau * (prob_state0**2) - 2 * tau * prob_state0 + tau)
                objective_wo_mult += obj - ex  # + LAMBDA * allocations[i]

                # Compute actual transfer
                midpoint = self.midpoints[i]
                transfer2 = midpoint * allocations[i]
                if allocations[i] >= 0:
                    transfer2 -= allocations[i]
                transfer2 -= gp.quicksum(allocations[:i] * self.step)

                # Compute actual externality
                externality2 = 1 - prob_state0 * (1 + allocations[i])
                if allocations[i] >= 0:
                    externality2 -= (1 - 2 * prob_state0) * allocations[i]
                externality2 *= tau * (1 - 2 * prob_state0)
                externality2 += tau * prob_state0

                avg_transfer += transfer2 * self.step * self._pdfs[i]
                avg_externality += externality2 * self.step * self._pdfs[i]
                continue

            virtual_value_neg = virtual_values.negative[i]
            virtual_value_pos = virtual_values.positive[i]
            LAMBDA = multi
            qs = np.linspace(-1, 1, 100)

            obj = (
                np.minimum(qs, 0) * virtual_value_neg
                + np.maximum(qs, 0) * virtual_value_pos
                - LAMBDA * qs
            )
            # if i == 80:
            #     # print(i, self.midpoints[i], np.argmax(obj))
            #     print(obj)
            allocations[i] = qs[np.argmax(obj)]

            ex = self._pdfs[i] * (2 * tau * (prob_state0**2) - 2 * tau * prob_state0 + tau)

            objective_value += np.max(obj) * self.step

            obj_wo_nult = (
                np.minimum(qs, 0) * virtual_value_neg + np.maximum(qs, 0) * virtual_value_pos
            ) + ex

            objective_wo_mult += obj_wo_nult[np.argmax(obj)]

            # Compute actual transfer
            midpoint = self.midpoints[i]
            transfer2 = midpoint * allocations[i]
            if allocations[i] >= 0:
                transfer2 -= allocations[i]
            transfer2 -= gp.quicksum(allocations[:i] * self.step)

            # Compute actual externality
            externality2 = 1 - prob_state0 * (1 + allocations[i])
            if allocations[i] >= 0:
                externality2 -= (1 - 2 * prob_state0) * allocations[i]
            externality2 *= tau * (1 - 2 * prob_state0)
            externality2 += tau * prob_state0

            avg_transfer += transfer2 * self.step * self._pdfs[i]
            avg_externality += externality2 * self.step * self._pdfs[i]

        transfers = self._allocations_to_transfers(allocations)
        externalities = self._allocations_to_externalities(allocations, tau, prob_state0)
        multiplier = LAMBDA  # model.getConstrByName("integral").Pi * self.num_intervals

        # print("objective_wo_mult", objective_wo_mult * self.step)

        return (
            allocations,
            transfers,
            externalities,
            multiplier,
            avg_transfer,
            avg_externality,
            objective_value,
            0,  # all_virtuals.X,
            0,  # binaries2.X,
        )
