from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field, model_validator


_FloatOrFloats = Union[float, np.ndarray[float]]


class Distribution(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass

    @abstractmethod
    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass


class Uniform(BaseModel, Distribution):
    """Uniform distribution defined by lower and upper bounds."""

    low: float = Field(..., description="Lower bound of the uniform distribution.")
    high: float = Field(..., description="Upper bound of the uniform distribution.")

    @model_validator(mode="after")
    def validate_bounds(self) -> "Uniform":
        if self.low >= self.high:
            raise ValueError("`low` must be less than `high`.")
        return self

    def __post_init__(self):
        self._dist = stats.uniform(loc=self.low, scale=self.high - self.low)

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.pdf(x)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.cdf(x)


class BetaMixture(BaseModel, Distribution):
    """Mixture of beta distributions with specified weights."""

    alphas: List[float] = Field(
        ..., description="List of alpha parameters for beta distributions."
    )
    betas: List[float] = Field(
        ..., description="List of beta parameters for beta distributions."
    )
    weights: List[float] = Field(
        ..., description="List of weights for the mixture components."
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> "BetaMixture":
        if len(self.alphas) != len(self.betas):
            raise ValueError("`alphas` and `betas` must have the same length.")
        if len(self.alphas) != len(self.weights):
            raise ValueError("`alphas` and `weights` must have the same length.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative.")
        return self

    def __post_init__(self):
        self._dists = [stats.beta(a, b) for a, b in zip(self.alphas, self.betas)]

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_pdfs = [d.pdf(x) * w for d, w in zip(self._dists, self.weights)]
        return np.sum(weighted_pdfs, axis=0)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_cdfs = [d.cdf(x) * w for d, w in zip(self._dists, self.weights)]
        return np.sum(weighted_cdfs, axis=0)
