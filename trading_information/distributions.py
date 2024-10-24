from typing import List
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, root_validator
import numpy as np
from scipy import stats

from trading_information.typing import _FloatOrFloats, _Floats


class Distribution(BaseModel, ABC):

    @abstractmethod
    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass

    @abstractmethod
    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass


class Uniform(BaseModel):
    """Uniform distribution defined by lower and upper bounds."""

    low: float
    high: float
    dist: stats.rv_continuous = Field(init=False)

    # Create distribution after initialization
    @root_validator(pre=False)
    def create_distribution(cls, values):
        low = values.get("low")
        high = values.get("high")
        values["dist"] = stats.uniform(loc=low, scale=high - low)
        return values

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self.dist.pdf(x)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self.dist.cdf(x)


class BetaMixture(BaseModel):
    """Mixture of beta distributions with specified weights."""

    alphas: _Floats
    betas: _Floats
    weights: _Floats
    dists: List[stats.rv_continuous] = Field(init=False)

    @root_validator(pre=False)
    def validate_and_create_distributions(cls, values):
        alphas = values.get("alphas")
        betas = values.get("betas")
        weights = values.get("weights")

        # Ensure lengths of alphas, betas, and weights are equal
        if not (len(alphas) == len(betas) == len(weights)):
            raise ValueError("Lengths of alphas, betas, and weights must be equal.")

        # Ensure the weights sum to 1
        sum_weights = np.sum(weights)
        if not np.abs(1 - sum_weights) <= 1e-9:
            raise ValueError(f"Mixture weights must sum to 1, got: {sum_weights}")

        # Create the list of beta distributions
        values["dists"] = [stats.beta(a, b) for a, b in zip(alphas, betas)]

        return values

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_pdfs = [d.pdf(x) * w for d, w in zip(self.dists, self.weights)]
        return np.sum(weighted_pdfs, axis=0)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_cdfs = [d.cdf(x) * w for d, w in zip(self.dists, self.weights)]
        return np.sum(weighted_cdfs, axis=0)
