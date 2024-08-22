from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from trading_information.typing import _FloatOrFloats, _Floats


class Distribution(ABC):

    @abstractmethod
    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass

    @abstractmethod
    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass


class Uniform(Distribution):
    """Uniform distribution defined by lower and upper bounds."""

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high
        self.distribution = stats.uniform(loc=low, scale=high - low)

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self.distribution.pdf(x)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self.distribution.cdf(x)


class BetaMixture(Distribution):
    """Mixture of beta distributions with specified weights."""

    def __init__(self, alphas: _Floats, betas: _Floats, weights: _Floats) -> None:
        if not (len(alphas) == len(betas) == len(weights)):
            raise ValueError("Lengths of alphas, betas, and weights must be equal.")

        if not np.abs(1 - (sum_weights := np.sum(weights))) <= 1e-9:
            raise ValueError(f"Mixture weights must sum to 1, got: {sum_weights}")

        self.alphas = alphas
        self.betas = betas
        self.weights = weights
        self.dists = [stats.beta(a, b) for a, b in zip(alphas, betas)]

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_pdfs = [dist.pdf(x) * weight for dist, weight in zip(self.dists, self.weights)]
        return np.sum(weighted_pdfs, axis=0)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_cdfs = [dist.cdf(x) * weight for dist, weight in zip(self.dists, self.weights)]
        return np.sum(weighted_cdfs, axis=0)
