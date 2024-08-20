from typing import Optional

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import numpy as np

from trading_information.distributions import Distribution
from trading_information.typing import _Floats, _FloatsCallable


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


class VirtualValues:
    """Calculate the (ironed) virtual values for a given type distribution"""

    def __init__(
        self,
        dist: Distribution,
        types: _Floats,
        alpha: Optional[float] = 0,
        iron: Optional[bool] = True,
    ) -> None:
        self.dist = dist
        self.types = types
        self.alpha = alpha
        self.iron = iron

        self.pdf = dist.pdf
        self.cdf = dist.cdf

        if self.iron:
            self.positive = self._ironed_positive_values()
            self.negative = self._ironed_negative_values()
        else:
            self.positive = self._unironed_positive_values()
            self.negative = self._unironed_negative_values()

    def _unironed_positive_values(self):
        return (self.types * (1 - self.alpha) - 1 + self.alpha) * self.pdf(self.types) + self.cdf(
            self.types
        )

    def _unironed_negative_values(self):
        return self.types * (1 + self.alpha) * self.pdf(self.types) + self.cdf(self.types)

    def _ironed_positive_values(self):
        integral = (self.types * (1 - self.alpha) - 1 + self.alpha) * self.cdf(
            self.types
        ) + self.alpha * cumulative_trapezoid(self.cdf(self.types), self.types, initial=0)

        # def integr(t):
        #     return (t * (1 - self.alpha) - 1 + self.alpha) * self.pdf(t) + self.cdf(t)

        # from scipy.integrate import quad

        # integral = np.zeros(len(self.types))
        # for i in range(len(integral)):
        #     t = self.types[i]
        #     integral[i] = quad(integr, a=0, b=t, limit=1000)[0]

        return self._iron(integral)

    def _ironed_negative_values(self):
        integral = self.types * (1 + self.alpha) * self.cdf(
            self.types
        ) - self.alpha * cumulative_trapezoid(self.cdf(self.types), self.types, initial=0)

        # def integr(t):
        #     return t * (1 + self.alpha) * self.pdf(t) + self.cdf(t)

        # from scipy.integrate import quad

        # integral = np.zeros(len(self.types))
        # for i in range(len(integral)):
        #     t = self.types[i]
        #     integral[i] = quad(integr, a=0, b=t, limit=1000)[0]

        # return np.zeros(len(integral))

        return self._iron(integral)

    def _iron(self, integral: _Floats) -> _Floats:
        envelope = convex_envelope(self.types, integral)

        return np.gradient(envelope, 1 / len(self.types))
