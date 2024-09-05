from typing import Callable, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from trading_information.typing import _Floats
from trading_information.distributions import Distribution


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
                    avg_transfer += transfer

                    # Update expression for average externality
                    externality = -increments[i] * prob_state0 * self._pdfs[i:].sum()
                    externality -= increments[i] * (1 - 2 * prob_state0) * self._pdfs[mask].sum()
                    avg_externality += externality

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

                model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
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
