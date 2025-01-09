from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import numpy as np
from pydantic import BaseModel, Field

from trading_information.distributions import Distribution

_Floats = np.ndarray[float]


def convex_envelope(xs: _Floats, ys: _Floats) -> _Floats:
    """Compute the convex envelope of the given function values at specified inputs.

    Args:
        xs (_Floats): Array of input values.
        ys (_Floats): Array of function values corresponding to the inputs.

    Returns:
        _Floats: Interpolated function representing the convex envelope.
    """
    xs_padded = np.pad(xs, (1, 1), mode="edge")
    ys_padded = np.pad(ys, (1, 1), constant_values=np.max(ys) + 1)

    # Create epigraph and compute its convex hull
    epigraph = np.column_stack((xs_padded, ys_padded))
    hull = ConvexHull(epigraph)
    envelope_indices = sorted(v - 1 for v in hull.vertices if 0 < v <= len(xs))

    return interp1d(xs[envelope_indices], ys[envelope_indices])(xs)


class VirtualValues(BaseModel):
    """Calculate the (possibly ironed) virtual values for a given type distribution."""

    dist: Distribution = Field(..., description="The distribution of types.")
    types: _Floats = Field(
        ..., description="Array of types for which to compute virtual values."
    )
    tau: float = Field(..., description="Parameter that encodes degree of competition.")
    prob_state0: float = Field(
        ..., description="The seller's own type (probability of state 0)."
    )
    iron: bool = Field(True, description="Whether to iron virtual values.")

    class Config:
        arbitrary_types_allowed = True

    @property
    def positive(self) -> _Floats:
        """Compute the positive virtual values."""
        return (
            self._ironed_positive_values()
            if self.iron
            else self._unironed_positive_values()
        )

    @property
    def negative(self) -> _Floats:
        """Compute the negative virtual values."""
        return (
            self._ironed_negative_values()
            if self.iron
            else self._unironed_negative_values()
        )

    @classmethod
    def get_ironed_values(
        cls, dist: Distribution, types: _Floats, tau: float, prob_state0: float
    ) -> tuple[_Floats, _Floats]:
        """Convenience method to get ironed virtual values.

        Args:
            dist (Distribution): The type distribution.
            types (_Floats): Array of types.
            tau (float): Tau parameter.
            prob_state0 (float): Probability of state 0.

        Returns:
            tuple[_Floats, _Floats]: Ironed positive and negative virtual values.
        """
        virtual_values = cls(
            dist=dist, types=types, tau=tau, prob_state0=prob_state0, iron=True
        )
        return virtual_values.positive, virtual_values.negative

    def _unironed_positive_values(self) -> _Floats:
        alpha = self.tau * (1 - 2 * self.prob_state0)
        return self.dist.pdf(self.types) * (
            self.types - 1 + alpha * (1 - self.prob_state0)
        ) + self.dist.cdf(self.types)

    def _unironed_negative_values(self) -> _Floats:
        alpha = self.tau * (1 - 2 * self.prob_state0)
        return self.dist.pdf(self.types) * (
            self.types + alpha * self.prob_state0
        ) + self.dist.cdf(self.types)

    def _ironed_positive_values(self) -> _Floats:
        alpha = self.tau * (1 - 2 * self.prob_state0)
        integral = self.dist.cdf(self.types) * (
            self.types + alpha * (1 - self.prob_state0) - 1
        )
        return self._do_the_ironing(integral)

    def _ironed_negative_values(self) -> _Floats:
        alpha = self.tau * (1 - 2 * self.prob_state0)
        integral = self.dist.cdf(self.types) * (self.types + alpha * self.prob_state0)
        return self._do_the_ironing(integral)

    def _do_the_ironing(self, integral: _Floats) -> _Floats:
        envelope = convex_envelope(self.types, integral)
        return np.gradient(envelope, self.types)
