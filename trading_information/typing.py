from typing import Union, Callable

import numpy as np

_Floats = np.ndarray[float]

_Ints = np.ndarray[int]

_FloatOrFloats = Union[float, _Floats]

_FloatsCallable = Callable[[_Floats], _Floats]
