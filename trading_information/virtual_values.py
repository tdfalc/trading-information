from typing import Optional

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import numpy as np
from pydantic import BaseModel

from trading_information.distributions import Distribution
from trading_information.typing import _Floats


def convex_envelope(xs: _Floats, ys: _Floats) -> _Floats:
    """Compute the convex envelope of the given function values at specified inputs.

    Args:
        xs (_Floats): Array of input values.
        ys (_Floats): Array of function values corresponding to the inputs.

    Returns:
        _Floats: Interpolated function representing the convex envelope.
    """
    xs_pad = np.pad(xs, (1, 1), mode="edge")
    ys_pad = np.pad(ys, (1, 1), constant_values=np.max(ys) + 1)

    # Create epigraph and compute its convex hull
    epi = np.column_stack((xs_pad, ys_pad))
    hull = ConvexHull(epi)
    envelope_indices = sorted(v - 1 for v in hull.vertices if 0 < v <= len(xs))

    return interp1d(x=xs[envelope_indices], y=ys[envelope_indices])(xs)


class VirtualValues(BaseModel):
    """Calculate the (possibly ironed) virtual values for a given type distribution."""

    dist: Distribution
    types: _Floats
    tau: float
    prob_state0: float
    iron: Optional[bool] = True

    # The positive and negative properties are computed based on whether `iron` is True
    @property
    def positive(self):
        return self._ironed_positive_values() if self.iron else self._unironed_positive_values()

    @property
    def negative(self):
        return self._ironed_negative_values() if self.iron else self._unironed_negative_values()

    @classmethod
    def get_ironed_values(cls, dist: Distribution, types: _Floats, tau: float, prob_state0: float):
        """Convenience function to simply get ironed virtual values."""
        virtual_values = cls(dist, types, tau, prob_state0, iron=True)
        return virtual_values.positive, virtual_values.negative

    def _unironed_positive_values(self):
        alpha = self.tau * (1 - 2 * self.prob_state0)
        return self.dist.pdf(self.types) * (
            self.types - 1 + alpha * (1 - self.prob_state0)
        ) + self.dist.cdf(self.types)

    def _unironed_negative_values(self):
        alpha = self.tau * (1 - 2 * self.prob_state0)
        return self.dist.pdf(self.types) * (self.types + alpha * self.prob_state0) + self.dist.cdf(
            self.types
        )

    def _ironed_positive_values(self):
        alpha = self.tau * (1 - 2 * self.prob_state0)
        integral = self.dist.cdf(self.types) * (self.types + alpha * (1 - self.prob_state0) - 1)
        return self._iron(integral)

    def _ironed_negative_values(self):
        alpha = self.tau * (1 - 2 * self.prob_state0)
        integral = self.dist.cdf(self.types) * (self.types + alpha * self.prob_state0)
        return self._iron(integral)

    def _iron(self, integral: _Floats) -> _Floats:
        envelope = convex_envelope(self.types, integral)
        return np.gradient(envelope, 1 / len(self.types))
