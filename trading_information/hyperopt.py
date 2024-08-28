from typing import Optional, Union
import functools
import contextlib

from joblib import Parallel, delayed
import numpy as np
import joblib
from tqdm import tqdm

from trading_information.mechanism import Mechanism
from trading_information.typing import _Floats


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

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


class HyperOpt:

    def __init__(
        self, mechanism: Mechanism, threshold_types: _Floats, n_jobs: Optional[int] = None
    ):
        self.mechanism = mechanism
        self.threshold_types = threshold_types
        self.n_jobs = n_jobs if n_jobs is not None else -1
        self._objectives = None

    @property
    def objectives(self):
        return self._objectives

    def run(self, tau: float, prob_state0: float, desc: Optional[str] = None) -> None:
        solve = functools.partial(self.mechanism.solve, tau=tau, prob_state0=prob_state0)
        with tqdm_joblib(tqdm(desc=desc, total=len(self.threshold_types))) as _:
            *_, self._objectives = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(solve)(threshold_type=threshold_type)
                    for threshold_type in self.threshold_types
                )
            )

    def best(self) -> float:
        idx = np.argmin(self._objectives)  # argmax in the other way
        return self.threshold_types[idx]
