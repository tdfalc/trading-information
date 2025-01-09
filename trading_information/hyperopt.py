from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable
import functools
import contextlib

from joblib import Parallel, delayed
import numpy as np
import joblib
from tqdm import tqdm
from pydantic import BaseModel, Field, PrivateAttr

from trading_information.mechanism import Mechanism

_Floats = np.ndarray[float]
_Ints = np.ndarray[int]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class HyperOptBase(BaseModel, ABC):
    """Base class for hyperparameter optimization."""

    mechanism: Mechanism = Field(
        ..., description="Mechanism instance for optimization."
    )
    n_jobs: int = Field(
        -1, description="Number of parallel jobs. Defaults to -1 (all available cores)."
    )
    verbose: bool = Field(True, description="Whether to display progress using tqdm.")
    _objectives: List[float] = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @property
    def objectives(self) -> Optional[List[float]]:
        return self._objectives

    def _run_parallel(
        self,
        solve_func: Callable[
            [Union[int, float]], float
        ],  # Callable accepting a single argument
        param_list: Union[_Ints, _Floats],
        desc: Optional[str],
    ) -> None:
        """Generic method for parallelized execution."""
        with (
            tqdm_joblib(tqdm(desc=desc, total=len(param_list)))
            if self.verbose
            else contextlib.nullcontext()
        ):
            *_, self._objectives = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(solve_func)(value) for value in param_list
                )
            )

    @abstractmethod
    def best(self) -> float:
        """Return the best parameter based on the optimization objectives."""
        pass

    @abstractmethod
    def run(self, tau: float, prob_state0: float, desc: Optional[str] = None) -> None:
        """Run the optimization."""
        pass


class HyperOptIncrements(HyperOptBase):
    """Optimization over threshold indices using increments."""

    threshold_indices: _Ints = Field(
        ..., description="Threshold indices to optimize over."
    )

    def run(self, tau: float, prob_state0: float, desc: Optional[str] = None) -> None:
        solve = functools.partial(
            self.mechanism.solve_with_increments, tau=tau, prob_state0=prob_state0
        )
        self._run_parallel(
            solve_func=solve, param_list=self.threshold_indices, desc=desc
        )

    def best(self) -> float:
        idx = np.argmax(self._objectives)
        return self.threshold_indices[idx]


class HyperOptVirtuals(HyperOptBase):
    """Optimization over multipliers using virtual values."""

    multipliers: _Floats = Field(..., description="Multipliers to optimize over.")

    def run(self, tau: float, prob_state0: float, desc: Optional[str] = None) -> None:
        solve = functools.partial(
            self.mechanism.solve_with_virtuals, tau=tau, prob_state0=prob_state0
        )
        self._run_parallel(solve_func=solve, param_list=self.multipliers, desc=desc)

    def best(self) -> float:
        idx = np.argmin(self._objectives)
        return self.multipliers[idx]
