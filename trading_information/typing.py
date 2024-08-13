from typing import Union, Callable

import numpy as np

_Floats = np.ndarray[float]

_FloatOrFloats = Union[float, _Floats]

_FloatsCallable = Callable[[_Floats], _Floats]
