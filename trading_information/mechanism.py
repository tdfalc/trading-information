from typing import Callable, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pydantic import BaseModel, Field, root_validator

from trading_information.typing import _Floats
from trading_information.distributions import Distribution
from trading_information.virtual_values import VirtualValues


class Mechanism(BaseModel):
    num_intervals: int
    dist: Distribution
    types: np.ndarray = Field(init=False)
    step: float = Field(init=False)
    midpoints: np.ndarray = Field(init=False)
    _pdfs: np.ndarray = Field(init=False)
    _cdfs: np.ndarray = Field(init=False)

    @root_validator(pre=False)
    def compute_derived_values(cls, values):
        num_intervals = values.get("num_intervals")
        dist = values.get("dist")

        types = np.linspace(0, 1, num_intervals + 1)
        step = 1 / num_intervals

        # Midpoints used to approximate integral with Riemann sum
        midpoints = (types[:-1] + types[1:]) / 2
        pdfs = dist.pdf(midpoints)
        cdfs = dist.cdf(midpoints)

        values["types"] = types
        values["step"] = step
        values["midpoints"] = midpoints
        values["_pdfs"] = pdfs
        values["_cdfs"] = cdfs

        return values


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
        self, allocations: _Floats, tau: float, prob_state0: float
    ) -> _Floats:

        prob_signal1 = (
            1
            - prob_state0
            - prob_state0 * allocations
            + (1 - 2 * prob_state0) * np.minimum(0, -allocations)
        )

        return prob_signal1 * (1 - 2 * prob_state0) * tau + tau * prob_state0

    def _convert_increments_to_allocations(self, increments: _Floats) -> _Floats:
        return np.cumsum(increments) - 1

    def solve_with_increments(self, tau: float, prob_state0: float, threshold_index: int):

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

    def solve_with_virtuals(
        self,
        tau: float,
        prob_state0: float,
        multiplier: float,
        integral_constraint: bool = False,
        pooling_constraint: bool = False,
    ):

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
                binaries = model.addMVar(self.num_intervals, vtype=gp.GRB.BINARY)
                binaries2 = model.addMVar(self.num_intervals, vtype=gp.GRB.BINARY)
                virtual_values = model.addMVar(self.num_intervals, lb=-np.inf, ub=np.inf)

                if integral_constraint:
                    model.addConstr(gp.quicksum(allocations) == 0)

                #  Get positive and negative ironed virtual values
                positive, negative = VirtualValues.get_ironed_values(
                    self.dist, self.midpoints, tau, prob_state0
                )

                # Lagrange objective not necessarily equal to difference between average transfer and average externality
                lagrangian, avg_transfer, avg_externality = 0, 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    model.addConstr(-allocations[i] <= 1 - binaries[i])
                    model.addConstr(allocations[i] <= binaries[i])

                    # Compute virtual vallue for allocation
                    virtual_value = negative[i] * (1 - binaries[i]) + positive[i] * binaries[i]
                    model.addConstr(virtual_values[i] == virtual_value)

                    ex = self._pdfs[i] * (2 * tau * (prob_state0**2) - 2 * tau * prob_state0 + tau)
                    lagrangian += (virtual_value - multiplier) * allocations[i] - ex

                    # Compute actual transfer
                    transfer = midpoint * allocations[i]
                    transfer -= allocations[i] * binaries[i]
                    transfer -= gp.quicksum(allocations[:i] * self.step)

                    # Compute actual externality
                    externality = 1 - prob_state0 * (1 + allocations[i])
                    externality -= (1 - 2 * prob_state0) * allocations[i] * binaries[i]
                    externality *= tau * (1 - 2 * prob_state0)
                    externality += tau * prob_state0

                    # Update average values
                    avg_transfer += transfer * self.step * self._pdfs[i]
                    avg_externality += externality * self.step * self._pdfs[i]

                    # M = 1000
                    # if i > 0 & pooling_constraint:
                    #     q = 1e-4
                    #     model.addConstr(
                    #         all_virtuals[i] - all_virtuals[i - 1] <= 1e-4 + M * (1 - binaries2[i])
                    #     )
                    #     model.addConstr(
                    #         all_virtuals[i] - all_virtuals[i - 1] >= 1e-4 - M * binaries2[i]
                    #     )
                    #     model.addGenConstrIndicator(
                    #         binaries2[i], 1, allocations[i] - allocations[i - 1] == 0
                    #     )

                model.setObjective(lagrangian * self.step, GRB.MAXIMIZE)
                model.optimize()

                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(allocations, tau, prob_state0)

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    avg_transfer.getValue(),
                    avg_externality.getValue(),
                    model.ObjVal,
                    binaries.X,
                    virtual_value.X,
                    binaries2.X,
                )
