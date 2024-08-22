from typing import Optional, Union
import functools

from joblib import Parallel, delayed
import numpy as np

from trading_information.mechanism import Mechanism
from trading_information.typing import _Floats


class HyperOpt:

    def __init__(
        self,
        mechanism: Mechanism,
        threshold_types: _Floats,
        n_jobs: Optional[int] = None,
        verbose: Optional[int] = None,
    ):
        self.mechanism = mechanism
        self.threshold_types = threshold_types
        self.n_jobs = n_jobs if n_jobs is not None else -1
        self.verbose = verbose if verbose is not None else 0
        self._objectives = None

    @property
    def objectives(self):
        return self._objectives

    def run(self, tau: float, prob_state0: float) -> None:
        solve = functools.partial(self.mechanism.solve, tau=tau, prob_state0=prob_state0)
        *_, self._objectives = zip(
            *Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(solve)(threshold_type=threshold_type)
                for threshold_type in self.threshold_types
            )
        )

    def best(self, tolerance: Optional[float] = None) -> Union[float, _Floats]:
        tolerance = 1e-9 if tolerance is None else tolerance
        diff = np.abs(self._objectives - np.max(self._objectives))
        indices = np.where(diff <= tolerance)[0]
        return [self.threshold_types[i] for i in indices]
