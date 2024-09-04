from typing import Optional, Union
import functools
import contextlib

from joblib import Parallel, delayed
import numpy as np
import joblib
from tqdm import tqdm

from trading_information.mechanism import Mechanism
from trading_information.typing import _Ints


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
        self, mechanism: Mechanism, threshold_indices: _Ints, n_jobs: Optional[int] = None
    ):
        self.mechanism = mechanism
        self.threshold_indices = threshold_indices
        self.n_jobs = n_jobs if n_jobs is not None else -1
        self._objectives = None

    @property
    def objectives(self):
        return self._objectives

    def run(self, tau: float, prob_state0: float, desc: Optional[str] = None) -> None:
        solve = functools.partial(self.mechanism.solve, tau=tau, prob_state0=prob_state0)
        with tqdm_joblib(tqdm(desc=desc, total=len(self.threshold_indices))) as _:
            *_, self._objectives = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(solve)(threshold_index=threshold_index)
                    for threshold_index in self.threshold_indices
                )
            )

    def best(self) -> float:
        idx = np.argmax(self._objectives)
        return self.threshold_indices[idx]
