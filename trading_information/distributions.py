from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field, model_validator, PrivateAttr


_FloatOrFloats = Union[float, np.ndarray[float]]


class Distribution(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass

    @abstractmethod
    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        pass


class Uniform(Distribution, BaseModel):
    """Uniform distribution defined by lower and upper bounds."""

    low: float = Field(..., description="Lower bound of the uniform distribution.")
    high: float = Field(..., description="Upper bound of the uniform distribution.")

    # Internal attribute for the scipy distribution
    _dist: stats.rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_initialize(self) -> "Uniform":
        if self.low >= self.high:
            raise ValueError("`low` must be less than `high`.")
        # Initialize the scipy uniform distribution
        object.__setattr__(
            self, "_dist", stats.uniform(loc=self.low, scale=self.high - self.low)
        )
        return self

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.pdf(x)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.cdf(x)


class Normal(Distribution, BaseModel):
    """Uniform distribution defined by lower and upper bounds."""

    loc: float = Field(..., description="Lower bound of the uniform distribution.")
    scale: float = Field(..., description="Upper bound of the uniform distribution.")

    # Internal attribute for the scipy distribution
    _dist: stats.rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_initialize(self) -> "Uniform":

        # Initialize the scipy uniform distribution
        object.__setattr__(self, "_dist", stats.norm(loc=self.loc, scale=self.scale))
        return self

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.pdf(x)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        return self._dist.cdf(x)


class BetaMixture(Distribution, BaseModel):
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

    # Internal attribute for the beta distributions
    _dists: List[stats.rv_continuous] = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_initialize(self) -> "BetaMixture":
        # Validate inputs
        if len(self.alphas) != len(self.betas):
            raise ValueError("`alphas` and `betas` must have the same length.")
        if len(self.alphas) != len(self.weights):
            raise ValueError("`alphas` and `weights` must have the same length.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative.")

        # Initialize the beta distributions
        object.__setattr__(
            self, "_dists", [stats.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        )
        return self

    def pdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_pdfs = [d.pdf(x) * w for d, w in zip(self._dists, self.weights)]
        return np.sum(weighted_pdfs, axis=0)

    def cdf(self, x: _FloatOrFloats) -> _FloatOrFloats:
        weighted_cdfs = [d.cdf(x) * w for d, w in zip(self._dists, self.weights)]
        return np.sum(weighted_cdfs, axis=0)
