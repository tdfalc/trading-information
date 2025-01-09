import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from trading_information.distributions import Distribution
from trading_information.virtual_values import VirtualValues

_Floats = np.ndarray[float]


class Mechanism(BaseModel):
    """Compute optimal mechanism for a given distribution of types."""

    num_intervals: int = Field(
        ..., gt=0, description="Number of intervals for integral approximation."
    )
    dist: Distribution = Field(..., description="The distribution of types.")

    _step: float = PrivateAttr()
    _midpoints: _Floats = PrivateAttr()
    _pdfs: _Floats = PrivateAttr()
    _cdfs: _Floats = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        types = np.linspace(0, 1, self.num_intervals + 1)
        self._step = 1 / self.num_intervals

        # Midpoints used to approximate integral with Riemann sum
        self._midpoints = (types[:-1] + types[1:]) / 2
        self._pdfs = self.dist.pdf(self._midpoints)
        self._cdfs = self.dist.cdf(self._midpoints)

    @property
    def midpoints(self) -> _Floats:
        """Return the midpoints used for Riemann sum approximation."""
        return self._midpoints

    def _allocations_to_transfers(self, allocations: _Floats) -> _Floats:
        return (
            self.midpoints * allocations
            - np.maximum(allocations, 0)
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

    def _increments_to_allocations(self, increments: _Floats) -> _Floats:
        return np.cumsum(increments) - 1

    def solve_with_increments(
        self,
        threshold_index: int,
        tau: float,
        prob_state0: float,
    ):
        """
        Solve the optimization problem using increments.

        Args:
            threshold_index (int): Index at which allocations change from negative to non-negative.
            tau (float): Parameter that encodes degree of competition.
            prob_state0 (float): The seller's own type (probability of state 0).

        Returns:
            tuple: Allocations, transfers, externalities, lagrange multiplier, and objective value.
        """
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                # Define decision variables
                increments = model.addMVar(self.num_intervals, lb=0)

                # Add constraints
                model.addConstr(
                    gp.quicksum(
                        increments * np.arange(self.num_intervals, 0, -1)
                    )
                    == self.num_intervals,
                    name="integral",
                )

                # These additional constraints ensure that the allocation function
                # is negative before the threshold index, and nonnegative afterwards.
                # model.addConstr(gp.quicksum(increments[:threshold_index]) >= 1)
                # model.addConstr(gp.quicksum(increments[threshold_index:]) >= 1)
                # model.addConstr(gp.quicksum(increments) <= 2)

                model.addConstr(gp.quicksum(increments[:threshold_index]) <= 1)
                model.addConstr(
                    gp.quicksum(increments[: threshold_index + 1]) >= 1
                )
                model.addConstr(gp.quicksum(increments) <= 2)

                # Initialize average transfer and externality
                avg_transfer = 0
                avg_externality = 0

                for i, midpoint in enumerate(self.midpoints):

                    threshold = self.midpoints[threshold_index]
                    mask = self.midpoints >= np.maximum(midpoint, threshold)

                    # Update expression for average transfer
                    # transfer = np.sum(
                    #     self.midpoints[i:] * self._pdfs[i:] + self._cdfs[i:]
                    # )
                    # transfer -= self._pdfs[mask].sum()
                    # transfer *= increments[i]

                    transfer = (
                        increments[i]
                        * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
                    )
                    transfer -= increments[i] * self._pdfs[mask].sum()
                    avg_transfer += transfer  # * self._pdfs[i]

                    avg_transfer += transfer

                    # Update expression for average externality
                    # externality = (2 * prob_state0 - 1) * self._pdfs[mask].sum()
                    # externality -= prob_state0 * self._pdfs[i:].sum()
                    # externality *= tau * (1 - 2 * prob_state0) * increments[i]

                    externality = (2 * prob_state0 - 1) * self._pdfs[mask].sum()
                    externality -= prob_state0 * self._pdfs[i:].sum()
                    externality *= tau * (1 - 2 * prob_state0) * increments[i]

                    avg_externality += externality

                # avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                # avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
                # avg_transfer *= self._step

                avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
                avg_transfer *= self._step

                # avg_externality += (
                #     tau
                #     * (1 - 2 * prob_state0)
                #     * (
                #         self._pdfs.sum()
                #         + (1 - 2 * prob_state0)
                #         * self._pdfs[self.midpoints >= threshold].sum()
                #     )
                # )
                # avg_externality += tau * prob_state0 * self.num_intervals

                # # avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                # #     self.midpoints >= threshold
                # # ].sum()
                # # avg_externality += self._pdfs.sum()
                # # avg_externality *= tau * (1 - 2 * prob_state0)
                # # avg_externality += tau * prob_state0 * self._pdfs.sum()
                # avg_externality *= self._step

                avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                    self.midpoints >= threshold
                ].sum()
                avg_externality += self._pdfs.sum()
                avg_externality *= tau * (1 - 2 * prob_state0)
                avg_externality += tau * prob_state0 * self._pdfs.sum()
                avg_externality *= self._step

                # Set objective function
                model.setObjective(avg_transfer - avg_externality, GRB.MAXIMIZE)
                model.optimize()

                # Extract results
                allocations = self._increments_to_allocations(increments.X)
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(
                    allocations, tau, prob_state0
                )
                multiplier = (
                    model.getConstrByName("integral").Pi * self.num_intervals
                )

                print(avg_transfer.getValue(), avg_externality.getValue())

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    model.ObjVal,
                )

    def solve_with_increments(
        self,
        threshold_index: int,
        tau: float,
        prob_state0: float,
    ):
        """
        Solve the optimization problem using increments.

        Args:
            threshold_index (int): Index at which allocations change from negative to non-negative.
            tau (float): Parameter that encodes degree of competition.
            prob_state0 (float): The seller's own type (probability of state 0).

        Returns:
            tuple: Allocations, transfers, externalities, lagrange multiplier, and objective value.
        """
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                # Define decision variables
                increments = model.addMVar(self.num_intervals, lb=0)
                model.addConstr(increments >= 0, name="monotonicity")
                model.addConstr(
                    gp.quicksum(
                        increments * np.arange(self.num_intervals, 0, -1)
                    )
                    == self.num_intervals,
                    name="integral",
                )

                # These additional constraints ensure that the allocation function
                # is negative before the threshold index, and nonnegative afterwards.
                model.addConstr(gp.quicksum(increments[:threshold_index]) <= 1)
                model.addConstr(
                    gp.quicksum(increments[: threshold_index + 1]) >= 1
                )
                model.addConstr(gp.quicksum(increments) <= 2)

                avg_transfer, avg_externality = 0, 0
                for i, midpoint in enumerate(self.midpoints):

                    threshold = self.midpoints[threshold_index]
                    mask = self.midpoints >= np.maximum(midpoint, threshold)

                    # Update expression for average transfer
                    transfer = (
                        increments[i]
                        * (self.midpoints * self._pdfs + self._cdfs)[i:].sum()
                    )
                    transfer -= increments[i] * self._pdfs[mask].sum()
                    avg_transfer += transfer  # * self._pdfs[i]

                    # Update expression for average externality
                    externality = (
                        -increments[i] * prob_state0 * self._pdfs[i:].sum()
                    )
                    externality -= (
                        increments[i]
                        * (1 - 2 * prob_state0)
                        * self._pdfs[mask].sum()
                    )
                    avg_externality += externality  # * self._pdfs[i]

                # These terms are constant with respect to the optimsation variables, however we
                # add them in we can sanity check the objective function value. Can delete.
                avg_transfer -= (self.midpoints * self._pdfs + self._cdfs).sum()
                avg_transfer += self._pdfs[self.midpoints >= threshold].sum()
                avg_transfer *= self._step

                avg_externality += (1 - 2 * prob_state0) * self._pdfs[
                    self.midpoints >= threshold
                ].sum()
                avg_externality += self._pdfs.sum()
                avg_externality *= tau * (1 - 2 * prob_state0)
                avg_externality += tau * prob_state0 * self._pdfs.sum()
                avg_externality *= self._step

                model.setObjective(
                    (avg_transfer - avg_externality), GRB.MAXIMIZE
                )
                model.optimize()

                allocations = self._increments_to_allocations(increments.X)
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(
                    allocations, tau, prob_state0
                )
                multiplier = (
                    model.getConstrByName("integral").Pi * self.num_intervals
                )

                print(avg_transfer.getValue(), avg_externality.getValue())

                return (
                    allocations,
                    transfers,
                    externalities,
                    multiplier,
                    model.ObjVal,
                )

    def solve_with_virtuals(
        self,
        tau: float,
        prob_state0: float,
        multiplier: float,
        integral_constraint: bool = False,
    ):
        """
        Solve the optimization problem using virtual values.

        Args:
            multiplier (float): Lagrangian multiplier for the integral constraint.
            tau (float): Parameter that encodes degree of competition.
            prob_state0 (float): The seller's own type (probability of state 0).
            integral_constraint (bool): Whether to enforce the integral constraint explicitly.

        Returns:
            tuple: Allocations, transfers, externalities, and objective value.
        """
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:

                # Define decision variables
                allocations = model.addMVar(self.num_intervals, lb=-1, ub=1)
                binaries = model.addMVar(
                    self.num_intervals, vtype=gp.GRB.BINARY
                )

                # Add integral constraint if specified
                if integral_constraint:
                    model.addConstr(gp.quicksum(allocations) == 0)

                # Get positive and negative ironed virtual values
                positive, negative = VirtualValues.get_ironed_values(
                    dist=self.dist,
                    types=self.midpoints,
                    tau=tau,
                    prob_state0=prob_state0,
                )

                # Initialize the lagrangian objective
                lagrangian = 0

                for i, _ in enumerate(self.midpoints):

                    # Add binary constraints to encode allocation sign
                    model.addGenConstrIndicator(
                        binaries[i],
                        True,
                        allocations[i],
                        GRB.GREATER_EQUAL,
                        1e-12,
                    )
                    model.addGenConstrIndicator(
                        binaries[i],
                        False,
                        allocations[i],
                        GRB.LESS_EQUAL,
                        0,
                    )

                    # Compute virtual value based on allocation sign
                    virtual_value = (
                        negative[i] * (1 - binaries[i])
                        + positive[i] * binaries[i]
                    )
                    lagrangian += (virtual_value - multiplier) * allocations[i]

                # Set the objective function
                model.setObjective(lagrangian * self._step, GRB.MAXIMIZE)
                model.optimize()

                # Extract results
                allocations = allocations.X
                transfers = self._allocations_to_transfers(allocations)
                externalities = self._allocations_to_externalities(
                    allocations, tau, prob_state0
                )

                return allocations, transfers, externalities, model.ObjVal
